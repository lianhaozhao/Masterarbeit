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

# DANN 的 λ 调度（只进 GRL）
def dann_lambda(epoch, num_epochs, max_lambda=0.5):
    p = epoch / max(1, num_epochs-1)
    return (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) * max_lambda

# LMMD 的基线权重（再乘质量门控得到最终权重）
def mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=5):
    if epoch < start_epoch: return 0.0
    p = (epoch-start_epoch) / max(1, (num_epochs-1-start_epoch))
    s = 1/(1+math.exp(-10*(p-0.5)))
    return float(max_lambda*s)

# ------------------ InfoMax（目标域） ------------------
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

# ------------------ 伪标签 + 统计 ------------------
@torch.no_grad()
def generate_pseudo_with_stats(model, target_loader, device, threshold=0.95, T=1.0):
    """
        在目标域无标签数据上，使用当前模型进行一次“离线伪标注”，并统计覆盖率与平均质量。

        该函数会对目标域的每个样本计算分类概率（可选温度 T），
        仅保留 Top-1 概率 conf >= threshold 的样本作为伪标签样本；
        同时用 (Top-1 - Top-2) 的概率差值作为“样本质量权重”（margin），
        以便后续在类条件对齐（如 LMMD）或蒸馏中对不确定样本降权。

        函数在 `torch.no_grad()` + `model.eval()` 下执行，不会产生梯度与参数更新。

        Parameters
        ----------
        model : torch.nn.Module
            已训练（或正在训练中的）分类模型。其 `forward(x, grl=False)` 应返回
            `(logits, domain_logits, features)`，其中本函数只使用 `logits`。
            注意：这里通过 `grl=False` 关闭 GRL 的反向影响（仅为语义明确；在 no_grad 下无反向）。
        target_loader : torch.utils.data.DataLoader
            目标域无标签数据的 DataLoader。`__getitem__` 应返回 `x` 或 `(x, ...)`，
            其中本函数仅使用第一个元素作为输入。
        device : torch.device
            推理设备（如 `torch.device("cuda")` 或 `torch.device("cpu")`）。
        threshold : float, default=0.95
            伪标签保留阈值：当样本的 Top-1 概率 `conf` 满足 `conf >= threshold` 时保留该样本。
            阈值越高，覆盖率（保留比例）通常越低但质量更高。
        T : float, default=1.0
            Softmax 温度。logits 会先除以 T 再做 softmax：
            - T > 1：概率更“平”（置信度下降，覆盖率通常降低）；
            - T < 1：概率更“尖”（置信度升高，覆盖率通常上升）。

        Returns
        -------
        x_cat : torch.Tensor
            通过阈值的目标样本拼接后的张量，位于 **CPU**。
        y_cat : torch.Tensor
            对应的“硬伪标签”（`argmax`），dtype 为 `torch.long`，位于 **CPU**。
        w_cat : torch.Tensor
            样本级权重（质量），使用 `margin = p_top1 - p_top2`，位于 **CPU**。
        stats : Dict[str, float]
            统计信息字典，包含：
            - "kept" : int，本轮保留的样本数 `N_keep`；
            - "total": int，目标域样本总数；
            - "coverage": float，覆盖率 = `N_keep / total`；
            - "margin_mean": float，保留样本的平均 margin，衡量伪标签整体质量。


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
        top2 = torch.topk(prob, k=2, dim=1).values #每个样本的 Top-1 概率 和 Top-2 概率 [B,2]
        conf, _ = torch.max(prob, dim=1)  #每个样本的Top-1 概率[B,1]
        margin = top2[:,0] - top2[:,1] #每个样本的 Top-1 概率 和 Top-2 概率差值
        keep = conf >= threshold
        if keep.any():
            xs.append(x[keep].cpu())
            ys.append(prob[keep].argmax(dim=1).cpu().long())
            # 用 margin 作为样本权重（质量）
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

#  加权类条件 MMD（多核 RBF）
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
    # MMD^2 = E_aa k + E_bb k - 2 E_ab k  （带权）
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

# 训练主循环
def train_dann_infomax_lmmd(model,
                            source_loader, target_loader,
                            optimizer, criterion_cls, criterion_domain,
                            device, num_epochs=20, num_classes=10,
                            pseudo_thresh=0.95,
                            mmd_gammas=(0.5,1,2,4,8),
                            scheduler=None, batch_size=16,
                            # InfoMax
                            im_T=1.0, im_weight=0.5, im_marg_w=1.0,
                            # 门控
                            lmmd_start_epoch=5,
                            ):
    best_state = None
    best_score = -1e18
    mmd_hist = deque(maxlen=5)
    gap_hist = deque(maxlen=5)

    for epoch in range(num_epochs):
        # 1) 伪标签
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

        # 2) 门控后的 LMMD 权重
        lambda_mmd_base = mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=lmmd_start_epoch)
        # 质量 q：对 margin 做线性归一（0.05~0.5），再与覆盖率耦合（concave，避免覆盖率高但质量差）
        def _lin(x, lo, hi):
            return float(min(max((x - lo) / max(1e-6, hi - lo), 0.0), 1.0))
        q_margin = _lin(margin_mean, 0.05, 0.50)
        q_cov = math.sqrt(max(0.0, cov))  # concave
        q = q_margin * q_cov
        lambda_mmd_eff = lambda_mmd_base * q

        # 3) epoch 训练
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
            # 把 λ 只放进 GRL
            model.lambda_ = float(dann_lambda(epoch, num_epochs))
            cls_out_src, dom_out_src, feat_src = model(src_x, grl=True)
            cls_out_tgt, dom_out_tgt, feat_tgt = model(tgt_x, grl=True)

            # 1) 源分类
            loss_cls = criterion_cls(cls_out_src, src_y)

            # 2) 域对抗
            dom_label_src = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            dom_label_tgt = torch.ones(tgt_x.size(0),  dtype=torch.long, device=device)
            loss_dom = (
                criterion_domain(dom_out_src, dom_label_src) * src_x.size(0)
                + criterion_domain(dom_out_tgt, dom_label_tgt) * tgt_x.size(0)
            ) / (src_x.size(0) + tgt_x.size(0))

            # 3) InfoMax
            loss_im, h_cond, h_marg = infomax_loss_from_logits(cls_out_tgt, T=im_T, marg_weight=im_marg_w)
            loss_im = im_weight * loss_im

            # 4) 类条件 LMMD（加权、质量门控）
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

            # ------ 统计 ------
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

        # ---- epoch 日志 ----
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

        mmd_hist.append(avg_mmd); gap_hist.append(gap)

        # print("[INFO] Evaluating on target test set...")
        # target_test_path = '../datasets/target/test/HC_T188_RP.txt'
        # test_dataset = PKLDataset(target_test_path)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # general_test_model(model, criterion_cls, test_loader, device)

    return model

# ------------------ main（示例） ------------------
if __name__ == "__main__":
    set_seed(44)
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['baseline']
    bs = cfg['batch_size']; lr = cfg['learning_rate']; wd = cfg['weight_decay']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = cfg['num_epochs']

    src_path = '../datasets/source/train/DC_T197_RP.txt'
    tgt_path = '../datasets/target/train/HC_T191_RP.txt'
    tgt_test = '../datasets/target/test/HC_T191_RP.txt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 10

    model = Flexible_DANN(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                          cnn_act='leakrelu', num_classes=NUM_CLASSES, lambda_=1.0).to(device)

    src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.1)
    c_cls = nn.CrossEntropyLoss(); c_dom = nn.CrossEntropyLoss()

    print("[INFO] Starting DANN + InfoMax + (quality-gated) LMMD ...")
    model = train_dann_infomax_lmmd(
        model, src_loader, tgt_loader,
        optimizer, c_cls, c_dom, device,
        num_epochs=num_epochs, num_classes=NUM_CLASSES,
        pseudo_thresh=0.95,
        scheduler=scheduler, batch_size=bs,
        # InfoMax 超参
        im_T=1.0, im_weight=0.5, im_marg_w=1.0,
        lmmd_start_epoch=5,
    )

    print("[INFO] Evaluating on target test set...")
    test_ds = PKLDataset(tgt_test)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    general_test_model(model, c_cls, test_loader, device)
