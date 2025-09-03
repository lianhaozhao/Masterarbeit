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
from models.generate_pseudo_labels_with_LMMD import pseudo_then_kmeans_select

def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_dataloaders(source_path, target_path, batch_size):
    src_ds = PKLDataset(txt_path=source_path)
    tgt_loader = get_target_loader(target_path, batch_size=batch_size, shuffle=True, drop_last=True)
    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return src_loader, tgt_loader

# DANN Lambda Scheduling (GRL only)
def dann_lambda(epoch, num_epochs, max_lambda=0.35):
    p = epoch / max(1, num_epochs-1)
    return (2.0 / (1.0 + np.exp(-5 * p)) - 1.0) * max_lambda

# Baseline weight of LMMD (multiplied by quality gate to get final weight)
def mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=5):
    if epoch < start_epoch: return 0.0
    p = (epoch-start_epoch) / max(1, (num_epochs-1-start_epoch))
    return (2.0 / (1.0 + np.exp(-7 * p)) - 1.0) * max_lambda

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
            Xs = feat_src[ms]             # [ns,d]
            Xt = feat_tgt[mt]             # [nt,d]
            wt = w_tgt[mt]                # [nt]

            # 协方差（源：无权；目标：带权）
            Cs = _weighted_cov(Xs, w=None, unbiased=unbiased_cov)
            Ct = _weighted_cov(Xt, w=wt,   unbiased=False)

            loss_c = coral_loss_from_cov(Cs, Ct, norm=cov_norm)

            if add_mean_align:
                mus = _weighted_mean(Xs, w=None)            # [1,d]
                mut = _weighted_mean(Xt, w=wt)              # [1,d]
                loss_mean = F.mse_loss(mus, mut, reduction='sum')  # 或 'mean' 再调系数
                loss_c = loss_c + mean_weight * loss_mean

            # 类权重
            w = float(min(ns, nt))
            total = total + loss_c * w
            wsum += w

    return total / wsum if wsum > 0 else total

# Training
def train_dann_infomax_lmmd(model,
                            source_loader, target_loader,
                            optimizer, criterion_cls, criterion_domain,
                            device, num_epochs=20, num_classes=10,
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
            pseudo_x, pseudo_y, pseudo_w, stats = pseudo_then_kmeans_select(
                model, target_loader, device, num_classes,
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

        # 2) Gated LMMD weights
        lambda_coral_eff = mmd_lambda(epoch, num_epochs, max_lambda=25e-2, start_epoch=lmmd_start_epoch)


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

            if tgt_pl_x is not None and lambda_coral_eff > 0:
                was_training = model.training
                model.eval()
                with torch.set_grad_enabled(True):
                    _, _, feat_src_lmmd = model(src_x, grl=False)
                    _, _, feat_tgt_lmmd = model(tgt_pl_x, grl=False)
                model.train(was_training)

                loss_coral = classwise_coral_weighted(
                    feat_src_lmmd, src_y, feat_tgt_lmmd, tgt_pl_y, tgt_pl_w,
                    num_classes=num_classes, min_count_per_class=3
                )
                loss_coral = lambda_coral_eff * loss_coral
            else:
                loss_coral = feat_src.new_tensor(0.0)

            loss = loss_cls + loss_dom + loss_im + loss_coral

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ------ Statistics ------
            cls_loss_sum  += loss_cls.item() * src_x.size(0)
            dom_loss_sum  += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            im_loss_sum += loss_im.item() * (tgt_x.size(0))
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
        avg_im  = im_loss_sum  / max(1, tot_target_samples)
        avg_coral = coral_loss_sum / max(1, tot_cls_samples)
        avg_tot = tot_loss_sum / max(1, tot_dom_samples)
        acc_src = dom_correct_src / max(1, dom_total_src)
        acc_tgt = dom_correct_tgt / max(1, dom_total_tgt)
        dom_acc = 0.5 * (acc_src + acc_tgt)
        gap = abs(dom_acc - 0.5)
        if scheduler is not None: scheduler.step()

        print(f"[Epoch {epoch+1}] Total loss:{avg_tot:.4f} | Avg cls loss:{avg_cls:.4f} | avg Dom loss:{avg_dom:.4f} "
              f"| avg IM loss:{avg_im:.4f} | avg coral loss:{avg_coral:.4f} | DomAcc:{dom_acc:.4f} | "
              f"cov:{cov:.2%} | "
              f"λ_GRL:{model.lambda_:.4f} | λ_coral_eff:{lambda_coral_eff:.4f}")

        gap_hist.append(gap)

        score = gap  + avg_im

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
        # print("[INFO] Evaluating on target test set...")
        # target_test_path = '../datasets/target/test/HC_T194_RP.txt'
        # test_dataset = PKLDataset(target_test_path)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # general_test_model(model, criterion_cls, test_loader, device)

    if plateau_best_state is not None:
        model.load_state_dict(plateau_best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    return model


if __name__ == "__main__":
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['baseline']
    bs = cfg['batch_size']
    lr = cfg['learning_rate']
    wd = cfg['weight_decay']
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
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.1)
            c_cls = nn.CrossEntropyLoss(); c_dom = nn.CrossEntropyLoss()

            print("[INFO] Starting DANN + InfoMax + (quality-gated) coral ...")
            model = train_dann_infomax_lmmd(
                model, src_loader, tgt_loader,
                optimizer, c_cls, c_dom, device,
                num_epochs=num_epochs, num_classes=10,
                pseudo_thresh=0.95,
                scheduler=scheduler, batch_size=bs,
                # InfoMax Hyperparameters
                im_T=1.0, im_weight=0.5, im_marg_w=1.0,
                lmmd_start_epoch=5, lmmd_t=2
            )

            print("[INFO] Evaluating on target test set...")
            test_ds = PKLDataset(tgt_test)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
            general_test_model(model, c_cls, test_loader, device)

            del model, optimizer, scheduler, src_loader, tgt_loader, test_loader, test_ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

