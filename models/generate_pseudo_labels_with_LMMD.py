import torch
import torch.nn.functional as F

def generate_pseudo_labels(model, target_loader, device, threshold=0.9, return_conf=True):
    """
    Generate pseudo-labels from unlabeled target data based on model predictions.

    Parameters:
        model : Trained model used for inference.
        target_loader (DataLoader): Dataloader providing unlabeled target domain data.
        threshold (float): Confidence threshold for accepting a pseudo-label.
        return_conf (bool): If True, also return per-sample confidence scores.

    Returns:
        pseudo_data (Tensor): Selected input samples with confidence >= threshold.
        pseudo_labels (Tensor): Corresponding pseudo-labels predicted by the model.
        pseudo_conf (Tensor, optional): Confidence values for each pseudo-label (if return_conf=True).
    """

    model.eval()
    pseudo_data, pseudo_labels, pseudo_conf = [], [], []

    with torch.no_grad():
        for inputs in target_loader:
            if isinstance(inputs, (tuple, list)):
                inputs = inputs[0]

            inputs = inputs.to(device)
            outputs = model(inputs)[0]  # 只取分类输出
            probs = F.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs, dim=1)

            mask = confidence >= threshold
            if mask.any():
                selected_inputs = inputs[mask]
                selected_labels = predicted[mask]
                selected_conf   = confidence[mask]

                pseudo_data.append(selected_inputs.cpu())
                pseudo_labels.append(selected_labels.cpu())
                pseudo_conf.append(selected_conf.cpu())

    if pseudo_data:
        pseudo_data = torch.cat(pseudo_data, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
        pseudo_conf = torch.cat(pseudo_conf, dim=0) if return_conf else None
    else:
        pseudo_data = torch.empty(0)
        pseudo_labels = torch.empty(0, dtype=torch.long)
        pseudo_conf = torch.empty(0)

    if return_conf:
        return pseudo_data, pseudo_labels, pseudo_conf
    else:
        return pseudo_data, pseudo_labels


import math, numpy as np, torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

# ========== 1) 提取“全部目标样本”的特征 + 伪标签 + margin(避免二次前向) ==========
@torch.no_grad()
def extract_all_target_feats_and_pseudo(model, target_loader, device, T=1.0, normalize=True):
    """
    对 target_loader 中的所有样本：提取 features、伪标签 y_hat、最大置信度 p_max、以及 top1-top2 margin。
    注意：这是对“全部样本”做的（不带阈值筛）。
    """
    model.eval()
    feats, y_hat, p_max, margins = [], [], [], []
    for batch in target_loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        logits, _, feat = model(x)  # <--- 若 model 不返回 feat，请在此改为你的取法
        prob = F.softmax(logits / T, dim=1)
        pm, yp = prob.max(dim=1)

        # 直接顺手算 margin，避免后面再跑一次网络
        top2 = torch.topk(prob, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1])

        if normalize:
            feat = F.normalize(feat, dim=1)

        feats.append(feat.detach().cpu())
        y_hat.append(yp.detach().cpu())
        p_max.append(pm.detach().cpu())
        margins.append(margin.detach().cpu())

    feats = torch.cat(feats, dim=0)            # [N, D], CPU
    y_hat = torch.cat(y_hat, dim=0).long()     # [N],  CPU
    p_max = torch.cat(p_max, dim=0)            # [N],  CPU
    margins_all = torch.cat(margins, dim=0)    # [N],  CPU
    return feats, y_hat, p_max, margins_all

# ========== 2) 全局 KMeans + 簇-类统计（修正 dtype/contiguous） ==========
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

# ========== 4) kNN 二次校验（源特征库；修正 dtype 一致性） ==========
@torch.no_grad()
def knn_second_pass(feats_sel_cpu, y_hat_sel_cpu, bank_feats_cpu, bank_labels_cpu, K=5, sim_th=0.20):
    if feats_sel_cpu is None or len(feats_sel_cpu) == 0:
        return torch.tensor([], dtype=torch.long)
    # 统一到 float32，避免隐式类型提升
    q = feats_sel_cpu.to(torch.float32)
    y = bank_feats_cpu.to(torch.float32)
    # 归一化假设已完成；内积作为余弦相似度
    sims = q @ y.T                                   # [M, Ns]
    vals, idx = torch.topk(sims, k=min(K, y.size(0)), dim=1)
    nbr_labels = bank_labels_cpu[idx]                # [M, K]
    # 多数票
    preds = []
    for r in range(nbr_labels.size(0)):
        labs, cnts = torch.unique(nbr_labels[r], return_counts=True)
        preds.append(labs[torch.argmax(cnts)])
    preds = torch.stack(preds).long()
    agree = (preds == y_hat_sel_cpu)
    good_sim = (vals[:, 0] >= sim_th)
    keep_mask = (agree & good_sim).nonzero(as_tuple=True)[0]
    return keep_mask

# ========== 5) 置信度阈值调度（保留原逻辑） ==========
def schedule_conf(epoch, start_ep=5, end_ep=20, hi=0.95, lo=0.90):
    if epoch <= start_ep: return hi
    if epoch >= end_ep:   return lo
    t = (epoch - start_ep) / max(1, (end_ep - start_ep))
    return float(hi + t*(lo - hi))

# ========== 6) 一站式：先伪标签 → 再 KMeans → 再筛选（+ 可选 kNN 校验） ==========
@torch.no_grad()
def pseudo_then_kmeans_select(
    model, target_loader, device, num_classes,
    epoch,
    # 伪标签/softmax
    T=1.0,
    # 调度后的置信度阈值
    conf_sched=(5, 20, 0.95, 0.90),
    # 聚类筛选阈值
    tau_pur=0.80, dist_q=0.80, min_cluster_size=20,
    per_class_cap=None,
    # 可选：kNN 二次校验
    bank_feats_cpu=None, bank_labels_cpu=None, knn_K=5, knn_sim_th=0.20
):
    # 1) 全量提取（含 margins，避免二次前向）
    feats_all, yhat_all, pmax_all, margins_all = extract_all_target_feats_and_pseudo(
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
        num_classes, tau_pur=tau_pur, conf_th=conf_th, dist_quantile=dist_q,
        min_cluster_size=min_cluster_size, per_class_cap=per_class_cap
    )

    # 5) 可选：kNN 二次校验（源库）
    if bank_feats_cpu is not None and bank_labels_cpu is not None and len(keep_idx) > 0:
        mask2 = knn_second_pass(
            feats_all[keep_idx], yhat_all[keep_idx],
            bank_feats_cpu, bank_labels_cpu,
            K=knn_K, sim_th=knn_sim_th
        )
        keep_idx = keep_idx[mask2]

    coverage = float(len(keep_idx) / max(1, feats_all.size(0)))
    log = {
        "cov_after_cluster": coverage,
        "conf_th": conf_th,
        "purity_mean": stat_cluster["purity_mean"],
        "purity_median": stat_cluster["purity_median"],
        "num_clusters_valid": stat_cluster["num_clusters_valid"],
        "median_pmax_all": float(pmax_all.median().item()),
        "mean_pmax_all": float(pmax_all.mean().item()),
    }

    # 直接用 yhat_all 和 margins_all 构造返回（避免二次前向）
    y_keep = yhat_all[keep_idx]          # 伪标签
    w_keep = margins_all[keep_idx]       # 质量权重(top1-top2)

    return keep_idx, y_keep, w_keep, log

