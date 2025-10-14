import os, matplotlib
matplotlib.use("Agg")  # 无显示环境时保存图片
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import numpy as np


# =========================
# 配置：类别名称与配色
# =========================
CLASS_NAMES = [f"R{5*i+5:02d}" for i in range(10)]  # ['R05','R10',...,'R50']

def cmap10():
    """使用 matplotlib 自带的高对比 10 色"""
    return plt.get_cmap("tab10")

# 若想用自定义鲜艳配色，注释上面行并启用下方：
VIVID10 = ListedColormap([
    "#E6194B", "#3CB44B", "#0082C8", "#F58231", "#911EB4",
    "#46F0F0", "#F032E6", "#D2F53C", "#FABEBE", "#008080",
])


# =========================
# 基础工具
# =========================
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


@torch.no_grad()
def collect_feats(model, loader, device, get_label: bool = True, pred_T: float = 1.0):
    """
    从 loader 收集类特征、标签/伪标签，以及目标域样本的置信度（margin）。
    返回:
      feats:  [N, d_feat]  —— forward 第三个返回（feat_c）
      labels: [N]          —— 源域用真标签，目标域用伪标签
      weights:[N]          —— 源域为 1；目标域为 softmax margin
    """
    model.eval()
    all_f, all_y, all_w = [], [], []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
            y = batch[1].to(device) if len(batch) > 1 and get_label else None
        else:
            x, y = batch.to(device), None

        logits, _, feat_c = model(x)  # 假定 forward 返回 (cls_out, dom_out, feat_c)

        prob = (logits / pred_T).softmax(dim=1)     # [B,C]
        conf, y_pred = prob.max(dim=1)              # 伪标签 & top1 概率
        top2 = prob.topk(2, dim=1).values[:, 1]     # 第二大概率
        margin = conf - top2                        # 置信边界

        all_f.append(feat_c.detach().cpu())
        if (y is not None) and get_label:
            all_y.append(y.detach().cpu())
            all_w.append(torch.ones_like(conf.detach().cpu()))
        else:
            all_y.append(y_pred.detach().cpu())
            all_w.append(margin.detach().cpu())

    feats = torch.cat(all_f, 0)
    labels = torch.cat(all_y, 0)
    weights = torch.cat(all_w, 0)
    return feats.numpy(), labels.numpy(), weights.numpy()


def js_divergence_2d(a: np.ndarray, b: np.ndarray, bins: int = 80) -> float:
    """
    用二维直方图近似计算 JS 散度（对称，>=0，越小越相似）
    a, b: [N,2]
    """
    eps = 1e-12
    mins = np.minimum(a.min(0), b.min(0))
    maxs = np.maximum(a.max(0), b.max(0))

    H_a, xedges, yedges = np.histogram2d(
        a[:, 0], a[:, 1], bins=bins,
        range=[[mins[0], maxs[0]], [mins[1], maxs[1]]],
        density=True
    )
    H_b, _, _ = np.histogram2d(
        b[:, 0], b[:, 1], bins=[xedges, yedges], density=True
    )

    P = (H_a + eps).ravel(); P = P / P.sum()
    Q = (H_b + eps).ravel(); Q = Q / Q.sum()
    M = 0.5 * (P + Q)

    KL_PM = np.sum(P * np.log(P / (M + eps) + eps))
    KL_QM = np.sum(Q * np.log(Q / (M + eps) + eps))
    return 0.5 * (KL_PM + KL_QM)


# =========================
# 可视化
# =========================
def plot_tsne_pca(feat_s: np.ndarray, y_s: np.ndarray,
                  feat_t: np.ndarray, y_t: np.ndarray,
                  save_path: str, title_prefix: str = "epoch",
                  use_vivid: bool = False):
    """
    绘制 t-SNE 与 PCA 2D 图，颜色表示类别；
    Source 域颜色更浅（淡化），Target 域更饱和；
    域与类别在图例中区分。
    """
    cmap = VIVID10 if use_vivid else cmap10()

    # ---- PCA 降维到 50 再做 t-SNE ----
    pca50 = PCA(n_components=min(50, feat_s.shape[1]))
    z_s = pca50.fit_transform(feat_s)
    z_t = pca50.transform(feat_t)

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                perplexity=30, max_iter=1000)
    z = tsne.fit_transform(np.vstack([z_s, z_t]))
    ns = z_s.shape[0]
    z_s2, z_t2 = z[:ns], z[ns:]

    # ==========================================================
    #  t-SNE 图
    # ==========================================================
    plt.figure(figsize=(9, 7))
    # Source 域（颜色更淡）
    plt.scatter(z_s2[:, 0], z_s2[:, 1],
                s=30, c=y_s, marker='o',
                cmap=cmap, vmin=0, vmax=9,
                alpha=0.45, edgecolors='none', label="Source")
    # Target 域（颜色饱和）
    plt.scatter(z_t2[:, 0], z_t2[:, 1],
                s=35, c=y_t, marker='^',
                cmap=cmap, vmin=0, vmax=9,
                alpha=0.8, edgecolors='black', linewidths=0.3, label="Target")

    # 图例
    present = np.unique(np.concatenate([y_s, y_t]).astype(int))
    class_handles = [
        plt.Line2D([0], [0], marker='s', linestyle='None',
                   color=cmap(i), label=CLASS_NAMES[i], markersize=8)
        for i in present
    ]
    domain_handles = [
        plt.Line2D([0], [0], marker='o', color='gray', linestyle='None',
                   label='Source ', markersize=7, alpha=0.5),
        plt.Line2D([0], [0], marker='^', color='gray', linestyle='None',
                   label='Target ', markersize=8, alpha=0.9)
    ]
    handles = domain_handles + class_handles
    plt.legend(handles=handles, frameon=True, ncol=5, fontsize=9,
               loc='best', title="Domains & Classes")

    plt.title(f"{title_prefix} | t-SNE", fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_tsne.png"), dpi=240)
    plt.close()

    # ==========================================================
    #  PCA 图
    # ==========================================================
    p2 = PCA(n_components=2)
    zs = p2.fit_transform(feat_s)
    zt = p2.transform(feat_t)

    plt.figure(figsize=(9, 7))
    plt.scatter(zs[:, 0], zs[:, 1],
                s=35, c=y_s, marker='o',
                cmap=cmap, vmin=0, vmax=9,
                alpha=0.45, edgecolors='none', label="Source")
    plt.scatter(zt[:, 0], zt[:, 1],
                s=55, c=y_t, marker='^',
                cmap=cmap, vmin=0, vmax=9,
                alpha=0.9, edgecolors='black', linewidths=0.3, label="Target")

    handles = domain_handles + class_handles
    plt.legend(handles=handles, frameon=True, ncol=5, fontsize=9,
               loc='best', title="Domains & Classes")

    plt.title(f"{title_prefix} | PCA", fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_pca.png"), dpi=240)
    plt.close()



def plot_class_center_heatmap(feat_s: np.ndarray, y_s: np.ndarray,
                              feat_t: np.ndarray, y_t: np.ndarray,
                              num_classes: int, save_path: str,
                              title: str = "Center Dist (1-cos)"):
    """
    计算源/目标每类中心（L2 归一化后求均值），
    以 1 - 余弦相似度 作为距离，绘制 CxC 热力图。
    返回：同类对角线距离的平均值（越小越对齐）
    """
    def class_centers(feat, y, C):
        f = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
        centers = np.zeros((C, f.shape[1]), dtype=np.float32)
        for c in range(C):
            idx = (y == c)
            if idx.sum() > 0:
                centers[c] = f[idx].mean(0)
        return centers

    Cs = class_centers(feat_s, y_s, num_classes)
    Ct = class_centers(feat_t, y_t, num_classes)

    Sim = Cs @ Ct.T                          # 余弦相似度（中心向量已单位化）
    D = 1.0 - np.clip(Sim, -1, 1)            # 距离矩阵

    plt.figure(figsize=(6, 5))
    plt.imshow(D, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.xlabel("Target class")
    plt.ylabel("Source class")
    ticks = np.arange(num_classes)
    plt.xticks(ticks, CLASS_NAMES[:num_classes], rotation=45, ha='right')
    plt.yticks(ticks, CLASS_NAMES[:num_classes])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

    diag = float(np.mean([D[i, i] for i in range(num_classes)]))
    return diag


def visualize_epoch(src_model, tgt_model, src_loader, tgt_loader,
                    device, num_classes: int, out_dir: str, epoch_tag: str,
                    pred_T: float = 1.0, use_vivid: bool = False,
                    conf_filter_quantile: float | None = None):
    """
    一次性生成：
      - t-SNE 图  ( *_tsne.png )
      - PCA  图   ( *_pca.png )
      - 类中心距离热力图 ( *_center_heatmap.png )
    返回：
      dict: { 'center_diag': float, 'js2d': float }
    参数：
      conf_filter_quantile: 可选，对目标域样本按 margin 过滤低置信（如 0.3 -> 丢弃最低 30%）
    """
    ensure_dir(out_dir)

    # 收集特征
    feat_s, y_s, _  = collect_feats(src_model, src_loader, device, get_label=True,  pred_T=pred_T)
    feat_t, y_t, w  = collect_feats(tgt_model, tgt_loader, device, get_label=False, pred_T=pred_T)

    # 可选：按置信度过滤目标域低置信样本，减少伪标签噪声
    if conf_filter_quantile is not None:
        q = np.quantile(w, conf_filter_quantile)
        keep = (w >= q)
        feat_t, y_t, w = feat_t[keep], y_t[keep], w[keep]

    # 可视化：t-SNE & PCA
    plot_tsne_pca(
        feat_s, y_s, feat_t, y_t,
        os.path.join(out_dir, f"{epoch_tag}_vis.png"),
        title_prefix=epoch_tag, use_vivid=use_vivid
    )

    # 类中心距离热力图
    diag = plot_class_center_heatmap(
        feat_s, y_s, feat_t, y_t, num_classes,
        os.path.join(out_dir, f"{epoch_tag}_center_heatmap.png"),
        title=f"{epoch_tag} | Center Dist (1-cos)"
    )

    # 分布相似度（统一到同一 PCA 2D 空间，再算 JS）
    p2 = PCA(n_components=2)
    as2 = p2.fit_transform(feat_s)
    at2 = p2.transform(feat_t)
    js = js_divergence_2d(as2, at2, bins=80)

    return {"center_diag": float(diag), "js2d": float(js)}
