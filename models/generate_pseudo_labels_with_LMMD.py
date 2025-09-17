import torch
from sqlalchemy import false
import math, numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans

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
        logits, _, _ = model(x_dev)
        prob = F.softmax(logits / T, dim=1)
        top2 = torch.topk(prob, k=2, dim=1).values  # [B,2]
        conf, _ = torch.max(prob, dim=1)  # [B]
        margin = top2[:, 0] - top2[:, 1]  # [B]
        keep = conf >= threshold
        if keep.any():
            xs.append(x_dev[keep].detach().cpu())
            ys.append(prob[keep].argmax(dim=1).detach().cpu().long())
            ws.append(margin[keep].detach().cpu())
            margins.append(margin[keep].detach().cpu())
    if len(xs) == 0:
        x_cat = torch.empty(0)
        y_cat = torch.empty(0, dtype=torch.long)
        w_cat = torch.empty(0)
        cov = 0.0
        margin_mean = 0.0
    else:
        x_cat = torch.cat(xs, dim=0)
        y_cat = torch.cat(ys, dim=0)
        w_cat = torch.cat(ws, dim=0)
        cov = float(x_cat.size(0)) / max(1, total)
        margin_mean = float(torch.cat(margins).mean())
    return x_cat, y_cat, w_cat, {"kept": int(x_cat.size(0)), "total": int(total),
                                 "coverage": cov, "margin_mean": margin_mean}


# ========== 1) 提取“全部目标样本”的特征 + 伪标签  ==========
@torch.no_grad()
def extract_all_target_feats_and_pseudo(model, target_loader, device, T=1.0, normalize=True, keep_inputs=True):
    """
    对 target_loader 中的所有样本：提取 features、伪标签 y_hat、最大置信度 p_max，
    并（可选）缓存原始输入 x（CPU），以便后续按索引切出 pseudo_x。
    注意：这是对“全部样本”做的（不带阈值筛）。
    """
    model.eval()
    xs, feats, y_hat, p_max = [], [], [], []
    for batch in target_loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device, non_blocking=True)
        logits, _, feat = model(x,grl = False)

        prob = F.softmax(logits / T, dim=1)
        pm, yp = prob.max(dim=1)

        if normalize:
            feat = F.normalize(feat, dim=1)

        feats.append(feat.detach().cpu())
        y_hat.append(yp.detach().cpu())
        p_max.append(pm.detach().cpu())
        if keep_inputs:
            xs.append(x.detach().cpu())

    feats = torch.cat(feats, dim=0)             # [N, D], CPU
    y_hat = torch.cat(y_hat, dim=0).long()      # [N],    CPU
    p_max = torch.cat(p_max, dim=0)             # [N],    CPU
    xs = torch.cat(xs, dim=0) if keep_inputs else None
    return xs, feats, y_hat, p_max
# ========== 1) 提取“全部目标样本”的特征 + 伪标签  ==========
@torch.no_grad()
def adda_extract_all_target_feats_and_pseudo(model, pseudo_loader, device, T=2.0, normalize=True, keep_inputs=True):
    """
    对 target_loader 中的所有样本：提取 features、伪标签 y_hat、最大置信度 p_max，
    并（可选）缓存原始输入 x（CPU），以便后续按索引切出 pseudo_x。
    注意：这是对“全部样本”做的（不带阈值筛）。
    """
    model.eval()
    xs, feats, y_hat, p_max = [], [], [], []
    for batch in pseudo_loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device, non_blocking=True)
        logits, _, feat = model(x)

        prob = F.softmax(logits / T, dim=1)
        pm, yp = prob.max(dim=1)

        if normalize:
            feat = F.normalize(feat, dim=1)

        feats.append(feat.detach().cpu())
        y_hat.append(yp.detach().cpu())
        p_max.append(pm.detach().cpu())
        if keep_inputs:
            xs.append(x.detach().cpu())

    feats = torch.cat(feats, dim=0)             # [N, D], CPU
    y_hat = torch.cat(y_hat, dim=0).long()      # [N],    CPU
    p_max = torch.cat(p_max, dim=0)             # [N],    CPU
    xs = torch.cat(xs, dim=0) if keep_inputs else None
    return xs, feats, y_hat, p_max
# ========== 2) 全局 KMeans + 簇-类统计 ==========
def kmeans_global(feats_cpu, num_classes, seed=0, n_init=10):
    # 确保 KMeans 输入是 float32 + contiguous
    x_np = feats_cpu.contiguous().to(torch.float32).numpy()
    km = KMeans(n_clusters=num_classes, n_init=n_init, random_state=seed)
    z_np = km.fit_predict(x_np)
    # centers 转回与 feats_cpu 相同 dtype，避免隐式类型提升
    centers = torch.from_numpy(km.cluster_centers_).to(feats_cpu.dtype)  # [C,D], CPU
    z = torch.from_numpy(z_np).long()                                    # [N], CPU
    return z, centers

@torch.no_grad()
def cluster_major_and_purity(y_hat, z, C):
    cluster_major = torch.full((C,), -1, dtype=torch.long)
    purity = torch.zeros(C, dtype=torch.float32)
    sizes = torch.zeros(C, dtype=torch.long)
    for k in range(C):
        idx = (z == k).nonzero(as_tuple=True)[0]
        sizes[k] = len(idx)
        if len(idx) == 0:
            continue
        labs, cnts = torch.unique(y_hat[idx], return_counts=True)
        j = torch.argmax(cnts)
        cluster_major[k] = labs[j]
        purity[k] = cnts[j].float() / len(idx)
    return cluster_major, purity, sizes

# ========== 3) 根据“簇纯度 + 距离分位 + 伪标签一致 + 置信度阈”筛选 ==========
@torch.no_grad()
def select_by_cluster_rules(
    feats_cpu, y_hat, p_max, z, centers, cluster_major, purity, sizes,
    num_classes,
    tau_pur=0.80,               # 突纯度阈值
    conf_th=0.95,               # 置信度阈值（可调度）
    dist_quantile=0.80,         # 距离分位阈（越小越靠近中心）
    min_cluster_size=20,        # 过滤极小簇
    per_class_cap=None          # 每类最多保留数（类平衡，可选）
):
    N = feats_cpu.size(0); C = num_classes
    centers_z = centers[z]                          # [N,D]
    d = torch.linalg.norm(feats_cpu - centers_z, dim=1)  # [N] 到各自簇心的 L2

    # 分簇计算距离分位阈
    dist_th = torch.zeros(C, dtype=feats_cpu.dtype)
    for k in range(C):
        idx = (z == k).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            dist_th[k] = 0
        elif len(idx) < 5:
            dist_th[k] = d[idx].max()
        else:
            dist_th[k] = torch.quantile(d[idx], dist_quantile)

    agree     = (y_hat == cluster_major[z])         # 伪标签 == 簇多数类
    good_pur  = (purity[z] >= tau_pur)              # 簇纯度达标
    good_size = (sizes[z] >= min_cluster_size)      # 簇大小达标
    good_dist = (d <= dist_th[z])                   # 距离靠中心
    good_conf = (p_max >= conf_th)                  # 置信度达标

    keep = agree & good_pur & good_size & good_dist & good_conf
    keep_idx = torch.where(keep)[0]                 # CPU 索引

    # 可选：类平衡（按 y_hat 分桶后限额；优先保留“近 + 稳”的样本）
    if (per_class_cap is not None) and (len(keep_idx) > 0):
        selected = []
        for c in range(C):
            idx_c = keep_idx[(y_hat[keep_idx] == c)]
            if len(idx_c) == 0:
                continue
            # 打分：距离越小越好 + 置信度越高越好
            score = 0.5*(1 - (d[idx_c] / (d[idx_c].max() + 1e-12))) + 0.5*p_max[idx_c]
            order = torch.argsort(score, descending=True)
            if per_class_cap > 0 and len(order) > per_class_cap:
                order = order[:per_class_cap]
            selected.append(idx_c[order])
        if selected:
            keep_idx = torch.cat(selected, 0)

    stats = {
        "purity_mean": float(purity[purity>0].mean().item()) if (purity>0).any() else 0.0,
        "purity_median": float(purity[purity>0].median().item()) if (purity>0).any() else 0.0,
        "num_clusters_valid": int(((purity >= tau_pur) & (sizes >= min_cluster_size)).sum().item()),
        "coverage_raw": float(len(keep_idx) / max(1, N))
    }
    return keep_idx, d, stats

# ========== 4) 置信度阈值调度（保留原逻辑） ==========
def schedule_conf(epoch, start_ep=5, end_ep=20, hi=0.95, lo=0.90):
    if epoch <= start_ep: return hi
    if epoch >= end_ep:   return lo
    t = (epoch - start_ep) / max(1, (end_ep - start_ep))
    return float(hi + t*(lo - hi))

# ========== 5) 一站式：返回 (pseudo_x, pseudo_y, pseudo_w, stats) ==========
@torch.no_grad()
def pseudo_then_kmeans_select(
    model, target_loader, device, num_classes,
    epoch,
    # 伪标签/softmax
    T=1.0,
    # 置信度阈值调度 (start_ep, end_ep, hi, lo)
    conf_sched=(5, 20, 0.95, 0.90),
    # 聚类筛选阈值
    tau_pur=0.80, dist_q=0.80, min_cluster_size=20,
    per_class_cap=None,
    # 是否启用基于“靠中心程度”和“簇纯度”的权重
    use_cluster_weight=True,
):
    """
    返回:
      pseudo_x: [M, ...]  通过筛选的目标域样本（CPU Tensor）
      pseudo_y: [M]       伪标签 (long, CPU)
      pseudo_w: [M]       每样本权重 (float, CPU, in [0,1])
      stats:    dict      统计信息
    依赖:
      - extract_all_target_feats_and_pseudo(model, loader, device, T, normalize=True)
        应返回: xs_all, feats_all, yhat_all, pmax_all  (均在 CPU，xs_all 是原始输入)
    """
    # 1) 全量提取（含原始输入，避免二次前向）
    xs_all, feats_all, yhat_all, pmax_all = extract_all_target_feats_and_pseudo(
        model, target_loader, device, T=T, normalize=True
    )

    # 2) 全局 KMeans(K=C)
    z, centers = kmeans_global(feats_all, num_classes, seed=0, n_init=10)
    cluster_major, purity, sizes = cluster_major_and_purity(yhat_all, z, num_classes)

    # 3) 动态置信度阈值
    conf_th = schedule_conf(epoch, *conf_sched)

    # 4) 样本级筛选（簇-类一致 + 纯度 + 距离 + 置信度）
    keep_idx, dist_all, stat_cluster = select_by_cluster_rules(
        feats_all, yhat_all, pmax_all, z, centers, cluster_major, purity, sizes,
        num_classes,
        tau_pur=tau_pur, conf_th=conf_th, dist_quantile=dist_q,
        min_cluster_size=min_cluster_size, per_class_cap=per_class_cap
    )

    # 5) 组装输出
    if keep_idx.numel() == 0:
        pseudo_x = torch.empty(0)
        pseudo_y = torch.empty(0, dtype=torch.long)
        pseudo_w = torch.empty(0)
    else:
        # 直接索引出样本与伪标签（均在 CPU）
        pseudo_x = xs_all[keep_idx]
        pseudo_y = yhat_all[keep_idx]

        # ---- 样本权重：置信度 × 簇纯度 × （越靠近中心越大）----
        w_conf = pmax_all[keep_idx].clone().to(torch.float32)

        if use_cluster_weight:
            d_keep = dist_all[keep_idx].clone()
            z_keep = z[keep_idx]

            # 在各自簇内做 min-max 归一化得到“靠中心程度”权重
            w_center = torch.zeros_like(d_keep, dtype=torch.float32)
            for k in range(num_classes):
                idx_k = (z_keep == k).nonzero(as_tuple=True)[0]
                if idx_k.numel() == 0:
                    continue
                dk = d_keep[idx_k]
                dmin, dmax = dk.min(), dk.max()
                if (dmax - dmin) < 1e-12:
                    w_center[idx_k] = 1.0
                else:
                    w_center[idx_k] = 1.0 - (dk - dmin) / (dmax - dmin)  # 越近中心→越大

            w_purity = purity[z_keep].to(torch.float32)
            pseudo_w = (w_conf * w_center * w_purity).clamp_(0.0, 1.0)
        else:
            pseudo_w = w_conf

    coverage = float(keep_idx.numel() / max(1, feats_all.size(0)))
    stats = {
        "cov_after_cluster": coverage,
        "conf_th": conf_th,
        "purity_mean": stat_cluster["purity_mean"],
        "purity_median": stat_cluster["purity_median"],
        "num_clusters_valid": stat_cluster["num_clusters_valid"],
        "median_pmax_all": float(pmax_all.median().item()),
        "mean_pmax_all": float(pmax_all.mean().item()),
        "num_selected": int(keep_idx.numel()),
    }

    return pseudo_x, pseudo_y, pseudo_w, stats

# ========== 5) 一站式：返回 (pseudo_x, pseudo_y, pseudo_w, stats) ==========
@torch.no_grad()
def adda_pseudo_then_kmeans_select(
    model, pseudo_loader, device, num_classes,
    epoch,
    # 伪标签/softmax
    T=1.0,
    # 置信度阈值调度 (start_ep, end_ep, hi, lo)
    conf_sched=(5, 20, 0.95, 0.90),
    # 聚类筛选阈值
    tau_pur=0.80, dist_q=0.80, min_cluster_size=20,
    per_class_cap=None,
    # 是否启用基于“靠中心程度”和“簇纯度”的权重
    use_cluster_weight=True,
):
    """
    返回:
      pseudo_x: [M, ...]  通过筛选的目标域样本（CPU Tensor）
      pseudo_y: [M]       伪标签 (long, CPU)
      pseudo_w: [M]       每样本权重 (float, CPU, in [0,1])
      stats:    dict      统计信息
    依赖:
      - extract_all_target_feats_and_pseudo(model, loader, device, T)
        应返回: xs_all, feats_all, yhat_all, pmax_all  (均在 CPU，xs_all 是原始输入)
    """
    # 1) 全量提取（含原始输入，避免二次前向）
    xs_all, feats_all, yhat_all, pmax_all = adda_extract_all_target_feats_and_pseudo(
        model, pseudo_loader, device, T=T, normalize=True
    )

    # 2) 全局 KMeans(K=C)
    z, centers = kmeans_global(feats_all, num_classes, seed=0, n_init=10)
    cluster_major, purity, sizes = cluster_major_and_purity(yhat_all, z, num_classes)

    # 3) 动态置信度阈值
    conf_th = schedule_conf(epoch, *conf_sched)

    # 4) 样本级筛选（簇-类一致 + 纯度 + 距离 + 置信度）
    keep_idx, dist_all, stat_cluster = select_by_cluster_rules(
        feats_all, yhat_all, pmax_all, z, centers, cluster_major, purity, sizes,
        num_classes,
        tau_pur=tau_pur, conf_th=conf_th, dist_quantile=dist_q,
        min_cluster_size=min_cluster_size, per_class_cap=per_class_cap
    )

    # 5) 组装输出
    if keep_idx.numel() == 0:
        pseudo_x = torch.empty(0)
        pseudo_y = torch.empty(0, dtype=torch.long)
        pseudo_w = torch.empty(0)
    else:
        # 直接索引出样本与伪标签（均在 CPU）
        pseudo_x = xs_all[keep_idx]
        pseudo_y = yhat_all[keep_idx]

        # ---- 样本权重：置信度 × 簇纯度 × （越靠近中心越大）----
        w_conf = pmax_all[keep_idx].clone().to(torch.float32)

        if use_cluster_weight:
            d_keep = dist_all[keep_idx].clone()
            z_keep = z[keep_idx]

            # 在各自簇内做 min-max 归一化得到“靠中心程度”权重
            w_center = torch.zeros_like(d_keep, dtype=torch.float32)
            for k in range(num_classes):
                idx_k = (z_keep == k).nonzero(as_tuple=True)[0]
                if idx_k.numel() == 0:
                    continue
                dk = d_keep[idx_k]
                dmin, dmax = dk.min(), dk.max()
                if (dmax - dmin) < 1e-12:
                    w_center[idx_k] = 1.0
                else:
                    w_center[idx_k] = 1.0 - (dk - dmin) / (dmax - dmin)  # 越近中心→越大

            w_purity = purity[z_keep].to(torch.float32)
            pseudo_w = (w_conf * w_center * w_purity).clamp_(0.0, 1.0)
        else:
            pseudo_w = w_conf

    coverage = float(keep_idx.numel() / max(1, feats_all.size(0)))
    stats = {
        "cov_after_cluster": coverage,
        "conf_th": conf_th,
        "purity_mean": stat_cluster["purity_mean"],
        "purity_median": stat_cluster["purity_median"],
        "num_clusters_valid": stat_cluster["num_clusters_valid"],
        "median_pmax_all": float(pmax_all.median().item()),
        "mean_pmax_all": float(pmax_all.mean().item()),
        "num_selected": int(keep_idx.numel()),
    }

    return pseudo_x, pseudo_y, pseudo_w, stats