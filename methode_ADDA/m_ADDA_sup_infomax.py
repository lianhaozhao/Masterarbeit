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
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_with_stats

def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# Baseline weight of LMMD (multiplied by quality gate to get final weight)
def sup_lambda(epoch, num_epochs, max_lambda=3e-1, start_epoch=5):
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



class Prototypes(nn.Module):
    """
    维护每个类别的原型向量（动量更新）。
    - proto: [C, d]，每类一个 d 维中心
    - momentum m ∈ [0,1): 越大表示更新更平滑
    """
    def __init__(self, num_classes, feat_dim, momentum=0.95):
        super().__init__()
        self.register_buffer('proto', torch.zeros(num_classes, feat_dim))
        self.m = float(momentum)

    @torch.no_grad()
    def update(self, feats, labels, weights=None):
        """
        使用当前批（通常是目标伪标样本）的特征更新原型。
        feats:   [N, d]
        labels:  [N]
        weights: [N] or None （可用置信度）
        """
        if feats.numel() == 0:
            return
        for c in labels.unique():
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            if weights is None:
                vec = feats[mask].mean(dim=0)
            else:
                w = weights[mask]
                vec = (w.unsqueeze(1) * feats[mask]).sum(dim=0) / (w.sum().clamp_min(1e-8))
            self.proto[c] = self.m * self.proto[c] + (1.0 - self.m) * vec

    def supcon_logits(self, feats, tau=0.1):
        """
        计算与原型的相似度 logits:  (normalize(f) @ normalize(proto)^T) / τ
        feats: [N, d]
        return: [N, C]
        """
        f = F.normalize(feats, dim=1, eps=1e-8)
        p = F.normalize(self.proto, dim=1, eps=1e-8)
        logits = f @ p.t()
        if tau is not None and tau > 0:
            logits = logits / float(tau)
        return logits

def classwise_supcon_loss(feat_tgt, y_tgt, proto_module, tau=0.1, weight=1.0):
    """
    用于训练步的监督对比损失（目标伪标签 vs 原型）。
    - 只对目标样本/伪标签计算；源域不需要。
    - 返回一个标量 loss，反向只改特征（原型用 EMA 更新，不反传）。
    """
    if feat_tgt.numel() == 0:
        return feat_tgt.new_tensor(0.0)
    logits = proto_module.supcon_logits(feat_tgt, tau=tau)   # [N, C]
    loss = F.cross_entropy(logits, y_tgt) * float(weight)
    return loss




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
def pretrain_source_classifier(
    src_model,
    source_loader,
    target_loader,
    optimizer,
    criterion_cls,
    device,
    num_epochs=5,
    scheduler=None,
    extra_target_pass=True  # 是否在每个 epoch 末额外跑一遍 target-only InfoMax
):
    """
    - 源域：交叉熵 -> 更新 feature_extractor + classifier
    - 目标域：InfoMax 仅更新 classifier（对 feature_extractor 用 eval()+no_grad 提特征）
    - 可选：epoch 末再做一遍 target-only InfoMax，扩大目标覆盖
    """
    src_model.train()
    tgt_iter = iter(target_loader)

    for epoch in range(num_epochs):
        tot_loss = tot_ce = tot_im = tot_n = 0.0

        # ===== 源域 CE + 目标域 InfoMax（只训分类头）的混合小循环 =====
        for xb, yb in source_loader:
            xb, yb = xb.to(device), yb.to(device)

            # 1) 源域监督 CE：正常 forward（允许更新提取器与分类头）
            logits_s, _, _ = src_model(xb)
            ce = criterion_cls(logits_s, yb)

            # 2) 目标域 InfoMax：只让分类头收到梯度
            try:
                batch_t = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(target_loader)
                batch_t = next(tgt_iter)
            x_t = batch_t[0] if isinstance(batch_t, (list, tuple)) else batch_t
            x_t = x_t.to(device)

            # 用 feature_extractor 提取 target 特征；不更新其参数与 BN 统计
            was_train = src_model.feature_extractor.training
            src_model.feature_extractor.eval()
            with torch.no_grad():
                feat_t = src_model.feature_extractor(x_t)
            src_model.feature_extractor.train(was_train)

            # 只通过分类头（classifier 的输入就是 feature_dim）
            logits_t_head = src_model.classifier(feat_t)
            L_im, Hc, Hm = infomax_loss_from_logits(logits_t_head,T = 1)

            loss = ce + 0.5 * L_im

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            tot_loss += loss.item() * bs
            tot_ce   += ce.item()   * bs
            tot_im   += L_im.item() * bs
            tot_n    += bs

        # ===== 额外 target-only InfoMax（只训分类头；可选）=====
        if extra_target_pass and epoch == num_epochs - 1:
            for batch_t in target_loader:
                x_t = batch_t[0] if isinstance(batch_t, (list, tuple)) else batch_t
                x_t = x_t.to(device)

                was_train = src_model.feature_extractor.training
                src_model.feature_extractor.eval()
                with torch.no_grad():
                    feat_t = src_model.feature_extractor(x_t)  # <<<<<< 同样先过 extractor
                src_model.feature_extractor.train(was_train)

                logits_t_head = src_model.classifier(feat_t)
                L_im_extra, _, _ = infomax_loss_from_logits(logits_t_head,T = 1)

                optimizer.zero_grad()
                L_im_extra.backward()
                optimizer.step()

        print(f"[SRC PRETRAIN+IM] Epoch {epoch+1}/{num_epochs} | "
              f"Loss:{tot_loss/max(1,tot_n):.4f} | CE:{tot_ce/max(1,tot_n):.4f} | "
              f"IM:{tot_im/max(1,tot_n):.4f}")

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
        _ , feat_s, feat_reduce= src_model(xb_s)
        feat_dim = feat_s.size(1)
        feat_reduce_dim = feat_reduce.size(1)
    D = DomainClassifier(feature_dim=feat_dim).to(device)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, weight_decay=weight_decay)
    c_dom = nn.CrossEntropyLoss().to(device)
    protos = Prototypes(num_classes=num_classes, feat_dim=feat_reduce_dim, momentum=0.95).to(device)
    sup_tau = 0.1

    # 4) 训练循环（交替优化 D 和 F_t）
    for epoch in range(num_epochs):
        # 准备伪标签以便 LMMD
        pl_loader = None
        pseudo_x = pseudo_y = pseudo_w = None
        cov = margin_mean = 0.0
        if epoch >= lmmd_start_epoch:
            pseudo_x, pseudo_y, pseudo_w, stats = generate_pseudo_with_stats(
                tgt_model, target_loader, device, threshold=0.90, T=2
            )
            cov, margin_mean = stats["coverage"], stats["margin_mean"]
            if pseudo_x.numel() > 0:
                pl_ds = TensorDataset(pseudo_x, pseudo_y, pseudo_w)
                pl_loader = DataLoader(pl_ds, batch_size=batch_size, shuffle=True)
        lambda_sup_base = sup_lambda(epoch, num_epochs, max_lambda=25e-2, start_epoch=lmmd_start_epoch)

        def _lin(x, lo, hi):
            return float(min(max((x - lo) / max(1e-6, hi - lo), 0.0), 1.0))
        q_margin = _lin(margin_mean, 0.05, 0.50)
        q_cov = math.sqrt(max(0.0, cov))  # concave
        q = q_margin * q_cov
        lambda_sub_eff = lambda_sup_base * q

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
        d_loss_sum = g_loss_sum = im_loss_sum = sup_loss_sum = ft_loss_sum = 0.0
        d_acc_sum = 0.0; d_cnt = 0

        for _ in range(steps//2):
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
                g_out = D(f_t)
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
                    _, _,f_t_pl = tgt_model(xpl)
                    # SupCon 原型对齐损失（替代 LMMD）
                    loss_sup = classwise_supcon_loss(
                        feat_tgt=f_t_pl, y_tgt=ypl, proto_module=protos, tau=sup_tau, weight=lambda_sub_eff
                    )
                else:
                    loss_sup = f_t.new_tensor(0.0)


                loss_ft = loss_g + loss_im + loss_sup
                opt_ft.zero_grad()
                loss_ft.backward()
                opt_ft.step()
                if it_pl is not None:
                    with torch.no_grad():
                        if wpl.numel() > 20:
                            cutoff = torch.quantile(wpl, 0.95).item()
                            wpl_clip = torch.clamp(wpl, max=cutoff)
                        else:
                            wpl_clip = wpl
                        protos.update(f_t_pl.detach(), ypl.detach(), weights=wpl_clip.detach())


                def to_scalar(x):
                    return x.detach().item() if torch.is_tensor(x) else float(x)

                g_loss_sum += to_scalar(loss_g)
                im_loss_sum += to_scalar(loss_im)
                sup_loss_sum += to_scalar(loss_sup)
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
            f"SUP:{sup_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"FT(total):{ft_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"D-acc:{d_acc_sum / max(1, d_cnt):.4f} | "
            f"cov:{cov:.2%} margin:{margin_mean:.3f} | lambda_sub_eff:{float(lambda_sub_eff):.4f} |"

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
    num_epochs = 20

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
                                      cnn_act='leakrelu', num_classes=10).to(device)

            src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
            pseudo_loader = get_pseudo_dataloaders(tgt_path, bs)
            optimizer_src = torch.optim.Adam(src_model.parameters(), lr=lr, weight_decay=wd)
            scheduler_src = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_src, T_max=num_epochs//2, eta_min=lr * 0.1)
            src_cls = nn.CrossEntropyLoss()

            print("[INFO] SRC pretrain (Fs + C) ...")
            pretrain_source_classifier(src_model, src_loader, tgt_loader, optimizer_src, src_cls, device,
                                       num_epochs=max(1, num_epochs // 2), scheduler=scheduler_src)

            # —— 阶段2：初始化目标编码器 Ft（从 Fs 拷贝），训练 ADDA（+可选IM/LMMD）
            tgt_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                      cnn_act='leakrelu', num_classes=10).to(device)
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
