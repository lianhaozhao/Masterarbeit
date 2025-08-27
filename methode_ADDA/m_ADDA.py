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
from models.get_no_label_dataloader import get_target_loader
from utils.general_train_and_test import general_test_model

def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_dataloaders(source_path, target_path, batch_size):
    src_ds = PKLDataset(txt_path=source_path)
    tgt_loader = get_target_loader(target_path, batch_size=batch_size, shuffle=True)
    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True)
    return src_loader, tgt_loader


# Baseline weight of LMMD (multiplied by quality gate to get final weight)
def mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=5):
    if epoch < start_epoch: return 0.0
    p = (epoch-start_epoch) / max(1, (num_epochs-1-start_epoch))
    s = 1/(1+math.exp(-10*(p-0.5)))
    return float(max_lambda*s)

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

# 伪标签 + 统计
@torch.no_grad()
def generate_pseudo_with_stats(model, target_loader, device, threshold=0.95, T=1.0):
    """
        Use the current model to perform "offline pseudo-annotation" on unlabeled data in the target domain, and calculate coverage and average quality.

        This function calculates the classification probability (with an optional temperature T) for each target domain sample.
        Only samples with a top-1 probability conf >= threshold are retained as pseudo-labeled samples.
        The difference in the (top-1 - top-2) probability is used as a "sample quality weight" (margin),
        to facilitate downweighting of uncertain samples in subsequent class-conditional alignment (such as LMMD) or distillation.

        This function is executed with `torch.no_grad()` + `model.eval()`, and no gradients or parameter updates are generated.

        Parameters
        ----------
        model : torch.nn.Module
        A trained (or training) classification model. Its `forward(x, grl=False)` should return
        `(logits, domain_logits, features)`. This function only uses `logits`.
        Note: GRL is disabled here with `grl=False` The reverse effect of `` (for semantic clarity only; no reverse effect with `no_grad`).

        target_loader : torch.utils.data.DataLoader
        DataLoader for the target domain's unlabeled data. `__getitem__` should return `x` or `(x, ...)`,
        where only the first element is used as input.

        device : torch.device
        The inference device (e.g., `torch.device("cuda")` or `torch.device("cpu")`).

        threshold : float, default=0.95
        The pseudo-label retention threshold: retain samples if their top-1 probability `conf` satisfies `conf >= threshold`.

        A higher threshold generally results in lower coverage (retention fraction) but higher quality.

        T : float, default=1.0
        Softmax temperature. Logits are divided by T before softmaxing:

        - T > 1: probabilities are flatter (decreasing confidence, generally lower coverage);

        - T < 1: Probability is more "sharp" (increasing confidence generally leads to higher coverage).

        Returns
        ------
        x_cat : torch.Tensor
        The concatenated tensor of target samples that pass the threshold, located in **CPU**.
        y_cat : torch.Tensor
        The corresponding "hard pseudo-label" (`argmax`), dtype `torch.long`, located in **CPU**.
        w_cat : torch.Tensor
        Sample-level weight (quality), using `margin = p_top1 - p_top2`, located in **CPU**.
        stats : Dict[str, float]
        Statistics dictionary, containing:
        - "kept" : int, the number of samples `N_keep` retained in this round;
        - "total" : int, the total number of target domain samples;
        - "coverage" : float, coverage = `N_keep / total`;
        - "margin_mean" : float, the average margin of the retained samples, a measure of the overall pseudo-label quality.

    """

    model.eval()
    xs, ys, ws = [], [], []
    margins = []
    total = 0
    for batch in target_loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        total += x.size(0)
        x_dev = x.to(device)
        logits, _ , _= model(x_dev)
        prob = F.softmax(logits / T, dim=1)
        top2 = torch.topk(prob, k=2, dim=1).values  # [B,2]
        conf, _ = torch.max(prob, dim=1)            # [B]
        margin = top2[:, 0] - top2[:, 1]            # [B]
        keep = conf >= threshold
        if keep.any():
            # ---- 修改处：用 GPU 上的掩码索引 GPU 张量，再搬回 CPU ----
            xs.append(x_dev[keep].detach().cpu())
            ys.append(prob[keep].argmax(dim=1).detach().cpu().long())
            ws.append(margin[keep].detach().cpu())
            margins.append(margin[keep].detach().cpu())
    if len(xs) == 0:
        x_cat = torch.empty(0); y_cat = torch.empty(0, dtype=torch.long); w_cat = torch.empty(0)
        cov = 0.0; margin_mean = 0.0
    else:
        x_cat = torch.cat(xs, dim=0)
        y_cat = torch.cat(ys, dim=0)
        w_cat = torch.cat(ws, dim=0)
        cov = float(x_cat.size(0)) / max(1, total)
        margin_mean = float(torch.cat(margins).mean())
    return x_cat, y_cat, w_cat, {"kept": int(x_cat.size(0)), "total": int(total),
                                 "coverage": cov, "margin_mean": margin_mean}


#  Weighted Class Conditional MMD (Multi-core RBF)
def _pairwise_sq_dists(a, b):
    # a: [m,d], b: [n,d]
    a2 = (a*a).sum(dim=1, keepdim=True)       # [m,1]
    b2 = (b*b).sum(dim=1, keepdim=True).t()   # [1,n]
    return a2 + b2 - 2 * (a @ b.t())

def _mk_kernel(a, b, gammas):
    d2 = _pairwise_sq_dists(a, b).clamp_min(0)
    k = 0.0
    for g in gammas:
        k = k + torch.exp(-g * d2)
    return k

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
    src_model, tgt_model, source_loader, target_loader,
    device, num_epochs=20, num_classes=10,batch_size=16,
    # 判别器/优化器
    lr_ft=1e-4, lr_d=1e-4, weight_decay=0.0,
    # InfoMax
    im_T=1.0, im_weight=0.5, im_marg_w=1.0,
    # Pseudo+LMMD
    lmmd_start_epoch=5, pseudo_thresh=0.95, mmd_gammas=(0.5,1,2,4,8), T_lmmd = 2
):
    # 1) 冻结源模型 (完全固定 F_s + C)
    src_model.eval(); freeze(src_model)

    # 2) 冻结目标模型的 classifier，只训练其 encoder
    for p in tgt_model.classifier.parameters():
        p.requires_grad = False
    enc_params = [p for n, p in tgt_model.named_parameters() if not n.startswith('classifier')]
    opt_ft = torch.optim.Adam(enc_params, lr=lr_ft, weight_decay=weight_decay)
    tgt_model.train()

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
        # 准备伪标签以便 LMMD（可选）
        pl_loader = None
        pseudo_x = pseudo_y = pseudo_w = None
        cov = margin_mean = 0.0
        if epoch >= lmmd_start_epoch:
            pseudo_x, pseudo_y, pseudo_w, stats = generate_pseudo_with_stats(
                tgt_model, target_loader, device, threshold=pseudo_thresh, T=T_lmmd
            )
            cov, margin_mean = stats["coverage"], stats["margin_mean"]
            if pseudo_x.numel() > 0:
                pl_ds = TensorDataset(pseudo_x, pseudo_y, pseudo_w)
                pl_loader = DataLoader(pl_ds, batch_size=batch_size, shuffle=True)
        lambda_mmd_base = mmd_lambda(epoch, num_epochs, max_lambda=2e-1, start_epoch=lmmd_start_epoch)
        def _lin(x, lo, hi):
            return float(min(max((x - lo) / max(1e-6, hi - lo), 0.0), 1.0))
        q_margin = _lin(margin_mean, 0.05, 0.50)
        q_cov = math.sqrt(max(0.0, cov))  # concave
        q = q_margin * q_cov
        lambda_mmd_eff = lambda_mmd_base * q

        it_src, it_tgt = iter(source_loader), iter(target_loader)
        it_pl = iter(pl_loader) if pl_loader is not None else None
        len_src, len_tgt = len(source_loader), len(target_loader)
        len_pl  = len(pl_loader) if pl_loader is not None else 0
        steps = max(len_src, len_tgt, len_pl) if len_pl > 0 else max(len_src, len_tgt)

        # 统计
        d_loss_sum = g_loss_sum = im_loss_sum = mmd_loss_sum = 0.0
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
            with torch.no_grad():
                _, f_s, _= src_model(xs)       # [B, d]
                _, f_t, _= tgt_model(xt)       # [B, d]
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

            # ---------- (B) 训练 F_t: min 交叉熵(D(F_t(xt)), “source”标签) ----------
            D.eval()
            for p in D.parameters():
                p.requires_grad = False
            logits_t, f_t, _  = tgt_model(xt)
            # logits_t = src_model.classifier(f_t)
            fool_lab = torch.ones(f_t.size(0), dtype=torch.long, device=device)  # 让 D 预测成“source” -1
            g_out = D(f_t)
            loss_g = c_dom(g_out, fool_lab)

            # InfoMax 正则 —— 更自信但不塌缩
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
                    f_s_n = F.normalize(f_s_n, dim=1)
                _, _,f_t_pl = tgt_model(xpl)
                f_t_pl_n = F.normalize(f_t_pl, dim=1)
                loss_lmmd = classwise_mmd_biased_weighted(
                    f_s_n, ys, f_t_pl_n, ypl, wpl,
                    num_classes=num_classes, gammas=mmd_gammas, min_count_per_class=2
                )
                loss_lmmd = lambda_mmd_eff * loss_lmmd
            else:
                loss_lmmd = f_t.new_tensor(0.0)

            loss_ft = loss_g + loss_im + loss_lmmd

            opt_ft.zero_grad()
            loss_ft.backward()
            opt_ft.step()
            for p in D.parameters():
                p.requires_grad = True
            D.train()
            # 统计
            d_loss_sum += loss_d.item()
            g_loss_sum += loss_g.item()
            im_loss_sum += (loss_im.item() if torch.is_tensor(loss_im) else float(loss_im))
            mmd_loss_sum += (loss_lmmd.item() if torch.is_tensor(loss_lmmd) else float(loss_lmmd))

        print(f"[ADDA] Epoch {epoch+1}/{num_epochs} | "
              f"D:{d_loss_sum/max(1,steps):.4f} | Ft:{g_loss_sum/max(1,steps):.4f} | "
              f"IM:{im_loss_sum/max(1,steps):.4f} | LMMD:{mmd_loss_sum/max(1,steps):.4f} | "
              f"D-acc:{(d_acc_sum/max(1,d_cnt)):.4f} | cov:{cov:.2%} margin:{margin_mean:.3f}"
              f"loss_lmmd:{loss_lmmd}")

        print("[INFO] Evaluating on target test set...")
        target_test_path = '../datasets/target/test/HC_T191_RP.txt'
        test_dataset = PKLDataset(target_test_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        general_test_model(tgt_model, c_dom, test_loader, device)

    return tgt_model

if __name__ == "__main__":
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    bs = cfg['batch_size']; lr = cfg['learning_rate']; wd = cfg['weight_decay']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = cfg['num_epochs']

    src_path = '../datasets/source/train/DC_T197_RP.txt'
    tgt_path = '../datasets/target/train/HC_T191_RP.txt'
    tgt_test = '../datasets/target/test/HC_T191_RP.txt'

    for run_id in range(5):
        print(f"\n========== RUN {run_id} (ADDA) ==========")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # —— 阶段1：建立并训练源域模型 Fs + C （复用 Flexible_DANN 但不使用其域头/GRL）
        src_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                  cnn_act='leakrelu', num_classes=10, lambda_=1.0).to(device)

        src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
        optimizer_src = torch.optim.Adam(src_model.parameters(), lr=lr, weight_decay=wd)
        scheduler_src = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_src, T_max=num_epochs, eta_min=lr * 0.1)
        src_cls = nn.CrossEntropyLoss()

        print("[INFO] SRC pretrain (Fs + C) ...")
        pretrain_source_classifier(src_model, src_loader, optimizer_src, src_cls, device,
                                   num_epochs=max(1, num_epochs//4), scheduler=scheduler_src)

        # —— 阶段2：初始化目标编码器 Ft（从 Fs 拷贝），训练 ADDA（+可选IM/LMMD）
        tgt_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                  cnn_act='leakrelu', num_classes=10, lambda_=0.0).to(device)
        copy_encoder_params(src_model, tgt_model, device)

        print("[INFO] ADDA stage (Ft vs D) + optional InfoMax/LMMD ...")
        tgt_model = train_adda_infomax_lmmd(
            src_model, tgt_model, src_loader, tgt_loader, device,
            num_epochs=num_epochs, num_classes=10,batch_size=bs,
            # 判别器/优化器
            lr_ft=lr, lr_d=lr, weight_decay=wd,
            # InfoMax
            im_T=1.0, im_weight=0.5, im_marg_w=1.0,
            # LMMD
            lmmd_start_epoch=5, pseudo_thresh=0.95, T_lmmd=2
        )


        print("[INFO] Evaluating on target test set...")
        test_ds = PKLDataset(tgt_test)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
        general_test_model(tgt_model, src_cls, test_loader, device)

        del src_model, tgt_model, optimizer_src, scheduler_src, src_loader, tgt_loader, test_loader, test_ds
        if torch.cuda.is_available(): torch.cuda.empty_cache()
