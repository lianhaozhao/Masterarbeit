import copy, math, random, os
import yaml
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.Flexible_DANN_LMMD import Flexible_DANN
from PKLDataset import PKLDataset
from models.get_no_label_dataloader import get_dataloaders
from utils.general_train_and_test import general_test_model
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_with_stats

def adam_param_groups(model, weight_decay):
    """对 BN/偏置不做 weight decay"""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

# DANN Lambda Scheduling (GRL only)
def dann_lambda(epoch, num_epochs, max_lambda=0.5):
    p = epoch / max(1, num_epochs-1)
    return (2.0 / (1.0 + np.exp(-5 * p)) - 1.0) * max_lambda

# Baseline weight of MMD (multiplied by quality gate to get final weight)
def mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=5):
    if epoch < start_epoch:
        return 0.0
    p = (epoch - start_epoch) / max(1, (num_epochs - 1 - start_epoch))
    return (2.0 / (1.0 + np.exp(-3 * p)) - 1.0) * max_lambda



def _standardize_feat(Z, eps=1e-6):
    mu = Z.mean(dim=0, keepdim=True)
    std = Z.std(dim=0, keepdim=True).clamp_min(eps)
    return (Z - mu) / std

def _shrink_cov(S, alpha=0.1, eps=1e-12):
    d = S.size(0)
    tr = torch.trace(S).clamp_min(eps)
    target = (tr / d) * torch.eye(d, dtype=S.dtype, device=S.device)
    return (1 - alpha) * S + alpha * target

def _weighted_mean(X, w=None, eps=1e-8):
    # X: [n,d], w: [n]
    if w is None:
        return X.mean(dim=0, keepdim=True)  # [1,d]
    w = w.view(-1, 1)
    s = torch.clamp(w.sum(), min=eps)
    return (w * X).sum(dim=0, keepdim=True) / s

def _weighted_cov(X, w=None, eps=1e-8, unbiased=False):
    """
    返回加权协方差矩阵 Σ: [d,d]
    X: [n,d], w: [n]
    """
    n, d = X.shape
    if n <= 1:
        return X.new_zeros(d, d)
    if w is None:
        mu = X.mean(dim=0, keepdim=True)                    # [1,d]
        Xc = X - mu
        # 无权：等价于 (Xc^T Xc) / (n-1)（若 unbiased）
        denom = (n - 1 if unbiased and n > 1 else n)
        return (Xc.t() @ Xc) / max(1.0, float(denom))
    else:
        w = w.view(-1, 1)                                   # [n,1]
        s = torch.clamp(w.sum(), min=eps)
        mu = (w * X).sum(dim=0, keepdim=True) / s           # [1,d]
        Xc = X - mu
        # 加权协方差：Xc^T diag(w) Xc / s   （常用定义；如需无偏修正可加权修正项）
        Sigma = (Xc.t() @ (w * Xc)) / s
        return Sigma

def coral_loss_from_cov(S_s, S_t, norm='fro', eps=1e-12):
    """
    标准 CORAL: || S_s - S_t ||_F^2
    可选归一化，避免维度/尺度差异影响权重。
    """
    D = S_s - S_t
    if norm == 'fro':
        return (D * D).sum()
    elif norm == 'trace':
        # trace 归一化：除以 d^2（或 4 d^2）等，常见实现里会做一个常数缩放
        d = S_s.size(0)
        return (D * D).sum() / (d * d + eps)
    else:
        return (D * D).sum()

def classwise_coral_weighted(
    feat_src, y_src, feat_tgt, y_tgt, w_tgt,
    num_classes,
    min_count_per_class=2,
    add_mean_align=True,         # 是否额外加入类均值对齐项
    mean_weight=1.0,             # 均值项系数
    cov_norm='trace',            # 'fro' 或 'trace'
    unbiased_cov=False           # 是否对无权情形使用 n-1
):
    """
    类条件 CORAL：对每个类 c，对齐源/目标的协方差（和可选的均值）。
    外层以 min(ns, nt) 对各类加权平均。
    目标端使用 w_tgt 作为样本权重。
    """
    total = feat_src.new_tensor(0.0)
    wsum = 0.0

    for c in range(num_classes):
        ms = (y_src == c)
        mt = (y_tgt == c)
        ns, nt = int(ms.sum()), int(mt.sum())
        if ns >= min_count_per_class and nt >= min_count_per_class:
            Xs0 = feat_src[ms]  # 原始特征（用于均值）
            Xt0 = feat_tgt[mt]
            wt = w_tgt[mt]

            # 1) 均值对齐（用原始特征）
            if add_mean_align:
                mus0 = _weighted_mean(Xs0, w=None)
                mut0 = _weighted_mean(Xt0, w=wt)
                loss_mean = F.mse_loss(mus0, mut0, reduction='sum')  # 或按维度归一化，见下
            else:
                loss_mean = 0.0

            # 2) 协方差对齐（用标准化后的特征，稳协方差）
            Xs = _standardize_feat(Xs0)
            Xt = _standardize_feat(Xt0)
            Cs = _weighted_cov(Xs, w=None, unbiased=unbiased_cov)
            Ct = _weighted_cov(Xt, w=wt, unbiased=False)

            alpha_s = float(min(0.2, 8.0 / max(1, Xs.size(0))))
            alpha_t = float(min(0.2, 8.0 / max(1, Xt.size(0))))
            Cs = _shrink_cov(Cs, alpha=alpha_s)
            Ct = _shrink_cov(Ct, alpha=alpha_t)

            loss_c = coral_loss_from_cov(Cs, Ct, norm=cov_norm)
            loss_c = loss_c + mean_weight * loss_mean

            # 类权重
            w = float(min(ns, nt))
            total = total + loss_c * w
            wsum += w

    return total / wsum if wsum > 0 else total

# Training
def train_dann_infomax_coral(model,
                             source_loader, target_loader,
                             optimizer, criterion_cls, criterion_domain,
                             device, num_epochs=20, num_classes=10,
                             scheduler=None, batch_size=16,
                             threshold = 0.5,T=2,coral_start_epoch=5,
                             # Gating
                             grl_lambda_max=1,max_lambda=0.5,
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
        if epoch >= coral_start_epoch:
            pseudo_x, pseudo_y, pseudo_w, stats = generate_pseudo_with_stats(
                model, target_loader, device, threshold=threshold, T=T
            )
            kept, total = stats["kept"], stats["total"]
            cov, margin_mean = stats["coverage"], stats["margin_mean"]
            if kept > 0:
                pl_ds = TensorDataset(pseudo_x, pseudo_y, pseudo_w)
                pl_loader = DataLoader(pl_ds, batch_size=batch_size, shuffle=True)

        # 2) Gated LMMD weights
        lambda_coral_eff = mmd_lambda(epoch, num_epochs, max_lambda=max_lambda, start_epoch=coral_start_epoch)


        # 3) epoch training
        model.train()
        it_src = iter(source_loader)
        it_tgt = iter(target_loader)
        it_pl  = iter(pl_loader) if pl_loader is not None else None
        len_src, len_tgt = len(source_loader), len(target_loader)
        len_pl = len(pl_loader) if pl_loader is not None else 0
        num_iters = max(len_src, len_tgt, len_pl) if len_pl > 0 else max(len_src, len_tgt)

        cls_loss_sum = dom_loss_sum = coral_loss_sum = im_loss_sum = 0.0
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
            model.lambda_ = float(dann_lambda(epoch, num_epochs, max_lambda=grl_lambda_max))
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


            # 3) Class-conditional LMMD (weighted, quality-gated)

            if tgt_pl_x is not None and lambda_coral_eff > 0:
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    _, _, feat_src_coral = model(src_x, grl=False)
                    _, _, feat_tgt_coral = model(tgt_pl_x, grl=False)
                model.train(was_training)
                loss_coral = classwise_coral_weighted(
                    feat_src_coral, src_y, feat_tgt_coral, tgt_pl_y, tgt_pl_w,
                    num_classes=num_classes, min_count_per_class=2
                )
                loss_coral = lambda_coral_eff * loss_coral
            else:
                loss_coral = feat_src.new_tensor(0.0)

            loss = loss_cls + loss_dom + loss_coral

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # ------ Statistics ------
            cls_loss_sum  += loss_cls.item() * src_x.size(0)
            dom_loss_sum  += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            coral_loss_sum  += loss_coral.item() * src_x.size(0)
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
        avg_coral = coral_loss_sum / max(1, tot_cls_samples)
        avg_tot = tot_loss_sum / max(1, tot_dom_samples)
        acc_src = dom_correct_src / max(1, dom_total_src)
        acc_tgt = dom_correct_tgt / max(1, dom_total_tgt)
        dom_acc = 0.5 * (acc_src + acc_tgt)
        gap = abs(dom_acc - 0.5)
        if scheduler is not None: scheduler.step()

        print(f"[Epoch {epoch+1}] Total loss:{avg_tot:.4f} | Avg cls loss:{avg_cls:.4f} | avg Dom loss:{avg_dom:.4f} "
              f"| avg coral loss:{avg_coral:.4f} | DomAcc:{dom_acc:.4f} | "
              f"cov:{cov:.2%} | "
              f"λ_GRL:{model.lambda_:.4f} | λ_coral_eff:{lambda_coral_eff:.4f}")

        gap_hist.append(gap)

        score = gap

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
                    elif best_state is not None:
                        model.load_state_dict(best_state)
                    break
            else:

                patience = 0

    if plateau_best_state is not None:
        model.load_state_dict(plateau_best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    return model


if __name__ == "__main__":
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['baseline']
    bs = 64
    lr = 0.000492917441228604
    wd = 4.032905396021506e-07
    num_layers = cfg['num_layers']
    ksz = cfg['kernel_size']
    sc = cfg['start_channels']
    num_epochs = cfg['num_epochs']

    files = [194]
    # files = [185, 188, 191, 194, 197]
    for file in files:

        src_path = '../datasets/source/train/DC_T197_RP.txt'
        tgt_path = '../datasets/target/train/HC_T{}_RP.txt'.format(file)
        tgt_test = '../datasets/target/test/HC_T{}_RP.txt'.format(file)

        print(f"[INFO] Loading HC_T{file} ...")

        for run_id in range(5):
            print(f"\n========== RUN {run_id} ==========")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Flexible_DANN(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                  cnn_act='leakrelu', num_classes=10, lambda_=1.0).to(device)

            src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
            optimizer = torch.optim.Adam(
                adam_param_groups(model, wd),
                lr=lr, betas=(0.9, 0.999)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.1)
            c_cls = nn.CrossEntropyLoss(); c_dom = nn.CrossEntropyLoss()

            print("[INFO] Starting DANN + InfoMax + (quality-gated) coral ...")
            model = train_dann_infomax_coral(
                model, src_loader, tgt_loader,
                optimizer, c_cls, c_dom, device,
                num_epochs=num_epochs, num_classes=10,
                scheduler=scheduler, batch_size=bs,
                threshold=0.95,T=1.5,coral_start_epoch=4,
                grl_lambda_max=0.7,max_lambda=0.5458,
            )

            print("[INFO] Evaluating on target test set...")
            test_ds = PKLDataset(tgt_test)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
            general_test_model(model, c_cls, test_loader, device)

            del model, optimizer, scheduler, src_loader, tgt_loader, test_loader, test_ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

