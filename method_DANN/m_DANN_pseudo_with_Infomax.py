import copy, math, random, os
import yaml
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.Flexible_DANN_pseudo_MMD import Flexible_DANN
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

# DANN Lambda Scheduling (GRL only)
def dann_lambda(epoch, num_epochs, max_lambda=0.5):
    p = epoch / max(1, num_epochs-1)
    return (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) * max_lambda

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
        logits, _, _ = model(x_dev, grl=False)
        prob = F.softmax(logits/T, dim=1)
        top2 = torch.topk(prob, k=2, dim=1).values #Top-1 probability and Top-2 probability of each sample [B,2]
        conf, _ = torch.max(prob, dim=1)  #Top-1 probability of each sample [B,1]
        margin = top2[:,0] - top2[:,1] #The difference between the top-1 probability and the top-2 probability of each sample
        keep = conf >= threshold
        if keep.any():
            xs.append(x[keep].cpu())
            ys.append(prob[keep].argmax(dim=1).cpu().long())
            # Use margin as sample weight (quality)
            ws.append(margin[keep].cpu())
            margins.append(margin[keep].cpu())
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

# Training
def train_dann_infomax_lmmd(model,
                            source_loader, target_loader,
                            optimizer, criterion_cls, criterion_domain,
                            device, num_epochs=20, num_classes=10,
                            pseudo_thresh=0.95,
                            mmd_gammas=(0.5,1,2,4,8),
                            scheduler=None, batch_size=16,
                            # InfoMax
                            im_T=1.0, im_weight=0.5, im_marg_w=1.0,
                            # Gating
                            lmmd_start_epoch=5,
                            ):
    # Hyperparameters related to early stopping
    W = 4
    GAP_TH = 0.05  # DomAcc distance threshold of 0.5 (the smaller the better alignment)
    PATIENCE = 3  # Stop after several rounds without improvement

    # Track Cache & Best Recording
    gap_hist = deque(maxlen=W)
    best_score = float('inf')
    best_state = None
    plateau_best_score = float('inf')
    plateau_best_state = None
    patience = 0

    for epoch in range(num_epochs):
        # 1) Pseudo-labeling
        pl_loader = None
        pseudo_x = torch.empty(0)
        pseudo_y = torch.empty(0, dtype=torch.long)
        pseudo_w = torch.empty(0)
        cov = margin_mean = 0.0
        if epoch >= lmmd_start_epoch:
            pseudo_x, pseudo_y, pseudo_w, stats = generate_pseudo_with_stats(
                model, target_loader, device, threshold=pseudo_thresh, T=1.0
            )
            kept, total = stats["kept"], stats["total"]
            cov, margin_mean = stats["coverage"], stats["margin_mean"]
            if kept > 0:
                pl_ds = TensorDataset(pseudo_x, pseudo_y, pseudo_w)
                pl_loader = DataLoader(pl_ds, batch_size=batch_size, shuffle=True)

        # 2) Gated LMMD weights
        lambda_mmd_base = mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=lmmd_start_epoch)
        # Quality q: Linearly normalize margin (0.05-0.5) and couple it with coverage (concave, to avoid high coverage but poor quality)
        def _lin(x, lo, hi):
            return float(min(max((x - lo) / max(1e-6, hi - lo), 0.0), 1.0))
        q_margin = _lin(margin_mean, 0.05, 0.50)
        q_cov = math.sqrt(max(0.0, cov))  # concave
        q = q_margin * q_cov
        lambda_mmd_eff = lambda_mmd_base * q

        # 3) epoch training
        model.train()
        it_src = iter(source_loader)
        it_tgt = iter(target_loader)
        it_pl  = iter(pl_loader) if pl_loader is not None else None
        len_src, len_tgt = len(source_loader), len(target_loader)
        len_pl = len(pl_loader) if pl_loader is not None else 0
        num_iters = max(len_src, len_tgt, len_pl) if len_pl > 0 else max(len_src, len_tgt)

        cls_loss_sum = dom_loss_sum = mmd_loss_sum = im_loss_sum = 0.0
        tot_loss_sum = 0.0
        tot_target_samples=tot_cls_samples = tot_dom_samples = 0
        dom_correct_src = dom_correct_tgt = 0
        dom_total_src = dom_total_tgt = 0

        for _ in range(num_iters):
            try: src_x, src_y = next(it_src)
            except StopIteration:
                it_src = iter(source_loader); src_x, src_y = next(it_src)
            try: tgt_x = next(it_tgt)
            except StopIteration:
                it_tgt = iter(target_loader); tgt_x = next(it_tgt)
            if isinstance(tgt_x, (tuple, list)): tgt_x = tgt_x[0]
            if it_pl is not None:
                try: tgt_pl_x, tgt_pl_y, tgt_pl_w = next(it_pl)
                except StopIteration:
                    it_pl = iter(pl_loader); tgt_pl_x, tgt_pl_y, tgt_pl_w = next(it_pl)
            else:
                tgt_pl_x = tgt_pl_y = tgt_pl_w = None

            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)
            if tgt_pl_x is not None:
                tgt_pl_x = tgt_pl_x.to(device)
                tgt_pl_y = tgt_pl_y.to(device)
                tgt_pl_w = tgt_pl_w.to(device)

            # forword
            # Put λ only into GRL
            model.lambda_ = float(dann_lambda(epoch, num_epochs))
            cls_out_src, dom_out_src, feat_src = model(src_x, grl=True)
            cls_out_tgt, dom_out_tgt, feat_tgt = model(tgt_x, grl=True)

            # 1) Source classification
            loss_cls = criterion_cls(cls_out_src, src_y)

            # 2) Domain confrontation
            dom_label_src = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            dom_label_tgt = torch.ones(tgt_x.size(0),  dtype=torch.long, device=device)
            loss_dom = (
                criterion_domain(dom_out_src, dom_label_src) * src_x.size(0)
                + criterion_domain(dom_out_tgt, dom_label_tgt) * tgt_x.size(0)
            ) / (src_x.size(0) + tgt_x.size(0))

            # 3) InfoMax
            loss_im, h_cond, h_marg = infomax_loss_from_logits(cls_out_tgt, T=im_T, marg_weight=im_marg_w)
            loss_im = im_weight * loss_im

            # 4) Class-conditional LMMD (weighted, quality-gated)
            feat_src_n = F.normalize(feat_src, dim=1)
            if tgt_pl_x is not None and lambda_mmd_eff > 0:
                _, _, feat_tgt_pl = model(tgt_pl_x, grl=False)
                feat_tgt_pl_n = F.normalize(feat_tgt_pl, dim=1)
                loss_lmmd = classwise_mmd_biased_weighted(
                    feat_src_n, src_y, feat_tgt_pl_n, tgt_pl_y, tgt_pl_w,
                    num_classes=num_classes, gammas=mmd_gammas, min_count_per_class=2
                )
                loss_lmmd = lambda_mmd_eff * loss_lmmd
            else:
                loss_lmmd = feat_src_n.new_tensor(0.0)

            loss = loss_cls + loss_dom + loss_im + loss_lmmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ------ Statistics ------
            cls_loss_sum  += loss_cls.item() * src_x.size(0)
            dom_loss_sum  += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            im_loss_sum   += loss_im.item()  * (tgt_x.size(0))
            mmd_loss_sum  += loss_lmmd.item() * src_x.size(0)
            tot_loss_sum  += loss.item()     * (src_x.size(0) + tgt_x.size(0))
            tot_cls_samples += src_x.size(0)
            tot_dom_samples += (src_x.size(0) + tgt_x.size(0))
            tot_target_samples += tgt_x.size(0)

            dom_correct_src += (dom_out_src.argmax(1) == dom_label_src).sum().item()
            dom_total_src   += dom_label_src.size(0)
            dom_correct_tgt += (dom_out_tgt.argmax(1) == dom_label_tgt).sum().item()
            dom_total_tgt   += dom_label_tgt.size(0)

        # ---- Epoch Log ----
        avg_cls = cls_loss_sum / max(1, tot_cls_samples)
        avg_dom = dom_loss_sum / max(1, tot_dom_samples)
        avg_im  = im_loss_sum  / max(1, tot_target_samples)
        avg_mmd = mmd_loss_sum / max(1, tot_cls_samples)
        avg_tot = tot_loss_sum / max(1, tot_dom_samples)
        acc_src = dom_correct_src / max(1, dom_total_src)
        acc_tgt = dom_correct_tgt / max(1, dom_total_tgt)
        dom_acc = 0.5 * (acc_src + acc_tgt)
        gap = abs(dom_acc - 0.5)
        if scheduler is not None: scheduler.step()

        print(f"[Epoch {epoch+1}] Total loss:{avg_tot:.4f} | Avg cls loss:{avg_cls:.4f} | avg Dom loss:{avg_dom:.4f} "
              f"| avg IM loss:{avg_im:.4f} | avg LMMD loss:{avg_mmd:.4f} | DomAcc:{dom_acc:.4f} | "
              f"cov:{cov:.2%} margin:{margin_mean:.3f} | "
              f"λ_GRL:{model.lambda_:.4f} | λ_mmd_eff:{lambda_mmd_eff:.4f}")

        gap_hist.append(gap)

        score =gap  + avg_im

        # Record the optimal model
        if epoch > 15:
            improved_global = (score < best_score - 1e-6)
            if improved_global:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

            gap_ok = (len(gap_hist) == W) and (sum(gap_hist) / W < GAP_TH)
            if gap_ok:
                improved_plateau = (score < plateau_best_score - 1e-6)
                if improved_plateau:
                    plateau_best_score = score
                    plateau_best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                print(f"[EARLY-STOP] patience {patience}/{PATIENCE} | gap_ok={gap_ok} | score={score:.4f}")
                if patience >= PATIENCE:
                    print("[EARLY-STOP] Stopping: stable alignment and no score improvement.")
                    # Prioritize backloading the "plateau optimal"; otherwise, backloading the "global optimal"
                    if plateau_best_state is not None:
                        model.load_state_dict(plateau_best_state)
                        torch.save(plateau_best_state, "./model/191_plateau_best.pth")
                    elif best_state is not None:
                        model.load_state_dict(best_state)
                        torch.save(best_state, "./model/191_best.pth")
                    break
            else:

                patience = 0
        print("[INFO] Evaluating on target test set...")
        target_test_path = '../datasets/target/test/HC_T194_RP.txt'
        test_dataset = PKLDataset(target_test_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        general_test_model(model, criterion_cls, test_loader, device)

    if plateau_best_state is not None:
        model.load_state_dict(plateau_best_state)
        torch.save(plateau_best_state, "./model/191_plateau_best.pth")
    elif best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, "./model/191_best.pth")

    return model


if __name__ == "__main__":
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['baseline']
    bs = cfg['batch_size']; lr = cfg['learning_rate']; wd = cfg['weight_decay']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = cfg['num_epochs']

    src_path = '../datasets/source/train/DC_T197_RP.txt'
    tgt_path = '../datasets/target/train/HC_T194_RP.txt'
    tgt_test = '../datasets/target/test/HC_T194_RP.txt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Flexible_DANN(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                          cnn_act='leakrelu', num_classes=10, lambda_=1.0).to(device)

    src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.1)
    c_cls = nn.CrossEntropyLoss(); c_dom = nn.CrossEntropyLoss()

    print("[INFO] Starting DANN + InfoMax + (quality-gated) LMMD ...")
    model = train_dann_infomax_lmmd(
        model, src_loader, tgt_loader,
        optimizer, c_cls, c_dom, device,
        num_epochs=num_epochs, num_classes=10,
        pseudo_thresh=0.95,
        scheduler=scheduler, batch_size=bs,
        # InfoMax Hyperparameters
        im_T=1.0, im_weight=0.5, im_marg_w=1.0,
        lmmd_start_epoch=5,
    )

    print("[INFO] Evaluating on target test set...")
    test_ds = PKLDataset(tgt_test)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    general_test_model(model, c_cls, test_loader, device)
