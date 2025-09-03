import copy, math, random, os
import yaml
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.Flexible_ADDA import Flexible_ADDA,freeze,unfreeze,DomainClassifier
from PKLDataset import PKLDataset
from models.get_no_label_dataloader import get_dataloaders, get_pseudo_dataloaders
from utils.general_train_and_test import general_test_model
from models.generate_pseudo_labels_with_LMMD import adda_pseudo_then_kmeans_select

def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# Baseline weight of LMMD (multiplied by quality gate to get final weight)
def mmd_lambda(epoch, num_epochs, max_lambda=3e-1, start_epoch=5):
    if epoch < start_epoch: return 0.0
    p = (epoch-start_epoch) / max(1, (num_epochs-1-start_epoch))
    return (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) * max_lambda

# ------------------ InfoMax (target domain) ------------------
@torch.no_grad()
def _safe_mean_prob(p, eps=1e-8):
    p = p.clamp_min(eps)
    return p / p.sum(dim=1, keepdim=True)

def entropy_mean(p, eps=1e-8):
    # E_x[ H(p_x) ]
    p = p.clamp_min(eps)
    return (-p * p.log()).sum(dim=1).mean()

def entropy_marginal(p, eps=1e-8):
    # H( E_x[p_x] )
    p_bar = p.mean(dim=0)
    p_bar = p_bar.clamp_min(eps)
    return -(p_bar * p_bar.log()).sum()

def infomax_loss_from_logits(logits, T=1.0, marg_weight=1.0):
    """
        依据 “信息最大化” (InfoMax) 思想，从分类 logits 计算无监督正则项：
        I(z; ŷ) = H(ŷ) - H(ŷ|z)。
        训练中最小化的目标为：
            L = H(ŷ|z) - w * H(ŷ)
        其中 H(ŷ|z) 是条件熵（鼓励单样本预测更自信），H(ŷ) 是边际熵（鼓励整体类别使用均衡，防止塌缩）。

        参数
        ----
        logits : Tensor
            分类头输出的未归一化得分，形状 [B, C]。
        T : float, 默认 1.0
            Softmax 温度。T > 1 使分布变“软”（置信度下降），T < 1 使分布变“尖”（置信度上升）。
            会同时影响条件熵与边际熵的数值。
        marg_weight : float, 默认 1.0
            边际熵权重 w。数值越大，越强烈地抑制“塌缩到单一类别”的解。
            常见范围 0.5 ~ 2.0，可据验证集曲线调参。

        返回
        ----
        loss : Tensor (标量，requires_grad=True)
            最小化目标：H(ŷ|z) - w * H(ŷ)。用于反向传播。
        h_cond_detached : Tensor (标量，no grad)
            条件熵 H(ŷ|z) 的经验估计（E_x[-∑_c p(c|x) log p(c|x)]）。仅用于日志监控。
        h_marg_detached : Tensor (标量，no grad)
            边际熵 H(ŷ) 的经验估计，其中 ŷ 的边际分布为 p̄ = E_x[p(·|x)]。
            数学形式：H(ŷ) = -∑_c p̄_c log p̄_c。仅用于日志监控。

        """
    # I(z;ŷ) = H(ŷ) - H(ŷ|z); minimiere  H(ŷ|z) - w * H(ŷ)
    p = F.softmax(logits / T, dim=1)  # [B, C], Klassenwahrscheinlichkeiten pro Beispiel
    h_cond = entropy_mean(p)  # Skalar, Schätzung der bedingten Entropie
    h_marg = entropy_marginal(p)  # Schätzung der marginalen Entropie
    return h_cond - marg_weight * h_marg, h_cond.detach(), h_marg.detach()



#  Weighted Class Conditional MMD (Multi-core RBF)
def _pairwise_sq_dists(a, b):
    # a: [m,d], b: [n,d]
    a2 = (a*a).sum(dim=1, keepdim=True)       # [m,1]
    b2 = (b*b).sum(dim=1, keepdim=True).t()   # [1,n]
    return a2 + b2 - 2 * (a @ b.t())

def _mk_kernel(a, b, gammas):
    d2 = _pairwise_sq_dists(a, b).clamp_min(0)
    k = 0.0
    M = max(1, len(gammas))
    for g in gammas:
        k = k + torch.exp(-float(g) * d2)
    return k / M

def _weighted_mean_kernel(K, w_row, w_col):
    # E_w[k] = (w_row^T K w_col) / (sum(w_row)*sum(w_col))
    num = (w_row.view(1,-1) @ K @ w_col.view(-1,1)).squeeze()
    den = (w_row.sum() * w_col.sum()).clamp_min(1e-8)
    return num / den

def mmd2_weighted(a, b, w_a=None, w_b=None, gammas=(0.5,1,2,4,8)):
    # MMD^2 = E_aa k + E_bb k - 2 E_ab k  （weight）
    if w_a is None: w_a = torch.ones(a.size(0), device=a.device)
    if w_b is None: w_b = torch.ones(b.size(0), device=b.device)
    Kaa = _mk_kernel(a, a, gammas)
    Kbb = _mk_kernel(b, b, gammas)
    Kab = _mk_kernel(a, b, gammas)
    e_aa = _weighted_mean_kernel(Kaa, w_a, w_a)
    e_bb = _weighted_mean_kernel(Kbb, w_b, w_b)
    e_ab = _weighted_mean_kernel(Kab, w_a, w_b)
    return (e_aa + e_bb - 2*e_ab).clamp_min(0.0)

@torch.no_grad()
def suggest_mmd_gammas(x_src, x_tgt, k=1024, scales=(0.25,0.5,1,2,4)):
    x = torch.cat([x_src.detach(), x_tgt.detach()], dim=0)
    n = x.size(0)
    if k is None or k >= n:
        xi, xj = x.unsqueeze(1), x.unsqueeze(0)
        d2 = (xi - xj).pow(2).sum(-1).flatten()
    else:
        i = torch.randint(0, n, (k,), device=x.device)
        j = torch.randint(0, n, (k,), device=x.device)
        d2 = (x[i]-x[j]).pow(2).sum(-1)
    m = d2.clamp_min(1e-12).median()
    g0 = (1.0 / (2.0 * m)).item()
    return [s * g0 for s in scales]

def classwise_mmd_biased_weighted(feat_src, y_src, feat_tgt, y_tgt, w_tgt,
                                  num_classes, gammas=(0.5,1,2,4,8),
                                  min_count_per_class=2):
    total = feat_src.new_tensor(0.0)
    wsum = 0.0
    for c in range(num_classes):
        ms = (y_src == c)
        mt = (y_tgt == c)
        ns, nt = int(ms.sum()), int(mt.sum())
        if ns >= min_count_per_class and nt >= min_count_per_class:
            w_c = w_tgt[mt]
            mmd_c = mmd2_weighted(feat_src[ms], feat_tgt[mt], None, w_c, gammas)
            w = float(min(ns, nt))
            total = total + mmd_c * w
            wsum += w
    return total / wsum if wsum > 0 else total





@torch.no_grad()
def copy_encoder_params(src_model, tgt_model, device):
    """
    依赖 Flexible_DANN 的 forward(x) 能返回 (logits, features)。
    我们不知道内部模块名，因此采取“整体复制”作为安全默认，然后在对抗阶段冻结 tgt_model 的 classifier，
    只训练其 encoder（因为 ADDA 要固定 C）。
    """
    tgt_model.load_state_dict(copy.deepcopy(src_model.state_dict()))
    tgt_model.to(device)


# ===== 新增：阶段1 —— 仅用源域训练 (F_s + C) =====
def pretrain_source_classifier(src_model, source_loader, optimizer, criterion_cls, device, num_epochs=5, scheduler=None):
    src_model.train()
    for epoch in range(num_epochs):
        tot_loss, tot_n = 0.0, 0
        for xb, yb in source_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _, _= src_model(xb)
            loss = criterion_cls(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * xb.size(0); tot_n += xb.size(0)
        print(f"[SRC PRETRAIN] Epoch {epoch+1}/{num_epochs} | cls:{tot_loss/max(1,tot_n):.4f}")
        if scheduler is not None:
            scheduler.step()
    return src_model

# ===== 新增：阶段2 —— ADDA 对抗 + (可选) InfoMax + (可选) LMMD =====
def train_adda_infomax_lmmd(
    src_model, tgt_model, source_loader, target_loader,pseudo_loader,
    device, num_epochs=20, num_classes=10,batch_size=16,
    # 判别器/优化器
    lr_ft=1e-4, lr_d=1e-4, weight_decay=0.0,d_steps =1 ,ft_steps =1 ,
    # InfoMax
    im_T=1.0, im_weight=0.5, im_marg_w=1.0,
    # Pseudo+LMMD
    lmmd_start_epoch=5
):
    # 1) 冻结源模型 (完全固定 F_s + C)
    src_model.eval(); freeze(src_model)

    # 2) 冻结目标模型的 classifier，只训练其 encoder
    for p in tgt_model.classifier.parameters():
        p.requires_grad = False
    tgt_model.classifier.eval()
    enc_params = list(tgt_model.feature_extractor.parameters()) \
                 + list(tgt_model.feature_reducer.parameters())
    opt_ft = torch.optim.Adam(enc_params, lr=lr_ft, weight_decay=weight_decay)

    # 3) 通过一个 batch 推断特征维度，按需构造 D
    with torch.no_grad():
        xb_s, yb_s = next(iter(source_loader))
        xb_s = xb_s.to(device)
        _ , feat_s, _= src_model(xb_s)
        feat_dim = feat_s.size(1)
    D = DomainClassifier(feature_dim=feat_dim).to(device)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, weight_decay=weight_decay)
    c_dom = nn.CrossEntropyLoss().to(device)

    # 4) 训练循环（交替优化 D 和 F_t）
    for epoch in range(num_epochs):
        # 准备伪标签以便 LMMD
        pl_loader = None
        pseudo_x = pseudo_y = pseudo_w = None
        cov = margin_mean = 0.0
        if epoch >= lmmd_start_epoch:
            pseudo_x, pseudo_y, pseudo_w, stats = adda_pseudo_then_kmeans_select(
                tgt_model, pseudo_loader, device, num_classes,
                epoch=epoch,
                T=1.0,
                conf_sched=(20, 30, 0.95, 0.90),  # 置信度阈值从 0.95 平滑降到 0.90
                tau_pur=0.80, dist_q=0.80, min_cluster_size=10,
                per_class_cap=None,  # 需要类平衡时，例如每类最多 500：per_class_cap=500
            )
            kept = stats["num_clusters_valid"]
            cov = stats["cov_after_cluster"]
            zahl = stats["num_selected"]
            if zahl > 0:
                pl_ds = TensorDataset(pseudo_x, pseudo_y, pseudo_w)
                pl_loader = DataLoader(pl_ds, batch_size=batch_size, shuffle=True)
        lambda_mmd_eff= mmd_lambda(epoch, num_epochs, max_lambda=25e-2, start_epoch=lmmd_start_epoch)

        it_src, it_tgt = iter(source_loader), iter(target_loader)
        it_pl = iter(pl_loader) if pl_loader is not None else None
        len_src, len_tgt = len(source_loader), len(target_loader)
        len_pl  = len(pl_loader) if pl_loader is not None else 0
        steps = max(len_src, len_tgt, len_pl) if len_pl > 0 else max(len_src, len_tgt)
        #新的迭代器
        it_tgt_ft = iter(target_loader)


        tgt_model.train()
        tgt_model.classifier.eval()
        D.train()

        # 统计
        d_loss_sum = g_loss_sum = im_loss_sum = mmd_loss_sum = ft_loss_sum = 0.0
        d_acc_sum = 0.0; d_cnt = 0

        for _ in range(steps):
            try: xs, ys = next(it_src)
            except StopIteration:
                it_src = iter(source_loader); xs, ys = next(it_src)
            try: xt = next(it_tgt)
            except StopIteration:
                it_tgt = iter(target_loader); xt = next(it_tgt)
            if isinstance(xt, (tuple,list)): xt = xt[0]
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            # ---------- (A) 训练 D: max log D(F_s(xs)) + log (1 - D(F_t(xt))) ----------
            for _k in range(d_steps):
                with torch.no_grad():
                    _, f_s, _= src_model(xs)       # [B, d]
                    _, f_t, _= tgt_model(xt)       # [B, d]
                    f_s = F.normalize(f_s, dim=1)  # 关键：L2 归一化
                    f_t = F.normalize(f_t, dim=1)
                d_in  = torch.cat([f_s, f_t], dim=0)
                d_lab = torch.cat([torch.ones(f_s.size(0)), torch.zeros(f_t.size(0))], dim=0).long().to(device)  # 1=source,0=target
                d_out = D(d_in)
                loss_d = c_dom(d_out, d_lab)
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                # 记录 D acc
                with torch.no_grad():
                    pred = d_out.argmax(1)
                    d_acc = (pred == d_lab).float().mean().item()
                    d_acc_sum += d_acc; d_cnt += 1
                    d_loss_sum += loss_d.item()

            # ---------- (B) 训练 F_t: min 交叉熵(D(F_t(xt)), “source”标签) ----------
            D.eval()
            for p in D.parameters():
                p.requires_grad = False

            for _k in range(ft_steps):
                try:
                    xt_ft = next(it_tgt_ft)
                except StopIteration:
                    it_tgt_ft = iter(target_loader)
                    xt_ft = next(it_tgt_ft)
                xt_ft = xt_ft.to(device)
                _, f_t, _  = tgt_model(xt_ft)
                fool_lab = torch.ones(f_t.size(0), dtype=torch.long, device=device)  # 让 D 预测成“source” -1
                f_t_n = F.normalize(f_t, dim=1)
                g_out = D(f_t_n)
                loss_g = c_dom(g_out, fool_lab)

                # InfoMax 正则 —— 更自信但不塌缩
                logits_t = src_model.classifier(f_t)
                loss_im, h_cond, h_marg = infomax_loss_from_logits(logits_t, T=im_T, marg_weight=im_marg_w)
                loss_im = im_weight * loss_im


                # LMMD：使用源真标签与目标伪标签做类条件对齐
                if it_pl is not None:
                    try: xpl, ypl, wpl = next(it_pl)
                    except StopIteration:
                        it_pl = iter(pl_loader); xpl, ypl, wpl = next(it_pl)
                    xpl, ypl, wpl = xpl.to(device), ypl.to(device), wpl.to(device)
                    with torch.no_grad():
                        _, _,f_s_n = src_model(xs)
                    _, _,f_t_pl = tgt_model(xpl)
                    gammas = suggest_mmd_gammas(f_s_n, f_t_pl, k=1024)
                    loss_lmmd = classwise_mmd_biased_weighted(
                        f_s_n, ys, f_t_pl, ypl, wpl,
                        num_classes=num_classes, gammas=gammas,
                        min_count_per_class=3
                    )
                    loss_lmmd = lambda_mmd_eff * loss_lmmd
                else:
                    loss_lmmd = f_t.new_tensor(0.0)


                loss_ft = loss_g + loss_im + loss_lmmd
                opt_ft.zero_grad()
                loss_ft.backward()
                opt_ft.step()

                def to_scalar(x):
                    return x.detach().item() if torch.is_tensor(x) else float(x)

                g_loss_sum += to_scalar(loss_g)
                im_loss_sum += to_scalar(loss_im)
                mmd_loss_sum += to_scalar(loss_lmmd)
                ft_loss_sum += to_scalar(loss_ft)
            for p in D.parameters():
                p.requires_grad = True
            D.train()


        # 打印
        print(
            f"[ADDA] Ep {epoch + 1}/{num_epochs} | "
            f"D:{d_loss_sum / max(1, steps * d_steps):.4f} | "
            f"G(adver):{g_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"IM:{im_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"LMMD:{mmd_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"FT(total):{ft_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"D-acc:{d_acc_sum / max(1, d_cnt):.4f} | "
            f"cov:{cov:.2%} margin:{margin_mean:.3f} | loss_lmmd:{float(loss_lmmd):.4f}"
        )

        print("[INFO] Evaluating on target test set...")
        target_test_path = '../datasets/target/test/HC_T194_RP.txt'
        test_dataset = PKLDataset(target_test_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        general_test_model(tgt_model, c_dom, test_loader, device)

    return tgt_model

if __name__ == "__main__":
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    bs = cfg['batch_size']; lr = cfg['learning_rate']; wd = cfg['weight_decay']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = 10

    files = [194]
    # files = [185, 188, 191, 194, 197]
    for file in files:
        src_path = '../datasets/source/train/DC_T197_RP.txt'
        tgt_path = '../datasets/target/train/HC_T{}_RP.txt'.format(file)
        tgt_test = '../datasets/target/test/HC_T{}_RP.txt'.format(file)

        print(f"[INFO] Loading HC_T{file} ...")

        for run_id in range(5):
            print(f"\n========== RUN {run_id} (ADDA) ==========")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # —— 阶段1：建立并训练源域模型 Fs + C （复用 Flexible_DANN 但不使用其域头/GRL）
            src_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                      cnn_act='leakrelu', num_classes=10, lambda_=1.0).to(device)

            src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
            pseudo_loader = get_pseudo_dataloaders(tgt_path, bs)
            optimizer_src = torch.optim.Adam(src_model.parameters(), lr=lr, weight_decay=wd)
            scheduler_src = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_src, T_max=num_epochs//2, eta_min=lr * 0.1)
            src_cls = nn.CrossEntropyLoss()

            print("[INFO] SRC pretrain (Fs + C) ...")
            pretrain_source_classifier(src_model, src_loader, optimizer_src, src_cls, device,
                                       num_epochs=max(1, num_epochs//2), scheduler=scheduler_src)

            # —— 阶段2：初始化目标编码器 Ft（从 Fs 拷贝），训练 ADDA（+可选IM/LMMD）
            tgt_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                      cnn_act='leakrelu', num_classes=10, lambda_=0.0).to(device)
            copy_encoder_params(src_model, tgt_model, device)

            print("[INFO] ADDA stage (Ft vs D) + optional InfoMax/LMMD ...")
            tgt_model = train_adda_infomax_lmmd(
                src_model, tgt_model, src_loader, tgt_loader,pseudo_loader, device,
                num_epochs=num_epochs, num_classes=10,batch_size=bs,
                # 判别器/优化器
                lr_ft=lr, lr_d=lr*0.5, weight_decay=wd,d_steps =1 ,ft_steps =2 ,
                # InfoMax
                im_T=1.0, im_weight=0.8, im_marg_w=1.0,
                # LMMD
                lmmd_start_epoch=3,
            )


            print("[INFO] Evaluating on target test set...")
            test_ds = PKLDataset(tgt_test)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
            general_test_model(tgt_model, src_cls, test_loader, device)

            del src_model, tgt_model, optimizer_src, scheduler_src, src_loader, tgt_loader, test_loader, test_ds
            if torch.cuda.is_available(): torch.cuda.empty_cache()
