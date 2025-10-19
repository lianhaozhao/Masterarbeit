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


def corporate_palette_10():
    """增强区分度的品牌配色 (10 类)"""
    base_rgb = [
        (0, 64, 112),     # 深蓝
        (0, 105, 170),    # 亮蓝（增强对比）
        (0, 159, 227),    # 天蓝
        (0, 180, 140),    # 青绿
        (126, 195, 153),  # 浅绿
        (201, 212, 0),    # 黄绿
        (253, 202, 0),    # 明黄
        (236, 97, 159),   # 洋红
        (255, 128, 64),   # 橙色
        (160, 90, 190),   # 紫色
    ]
    colors = [(r/255, g/255, b/255) for r, g, b in base_rgb]
    return ListedColormap(colors)


# =========================
# 工具函数
# =========================
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


@torch.no_grad()
def collect_feats(model, loader, device, get_label: bool = True, pred_T: float = 1.0):
    """从 DataLoader 收集特征、标签及置信度边界"""
    model.eval()
    all_f, all_y, all_w = [], [], []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
            y = batch[1].to(device) if len(batch) > 1 and get_label else None
        else:
            x, y = batch.to(device), None

        logits, _, feat_c = model(x)
        prob = (logits / pred_T).softmax(dim=1)
        conf, y_pred = prob.max(dim=1)
        top2 = prob.topk(2, dim=1).values[:, 1]
        margin = conf - top2

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
    """二维直方图近似 JS 散度"""
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
# 可视化函数
# =========================
def plot_tsne_pca(feat_s, y_s, feat_t, y_t, save_path, title_prefix="epoch"):
    """绘制 t-SNE 和 PCA 可视化图"""
    cmap = corporate_palette_10()

    # PCA 降维后再做 t-SNE
    pca50 = PCA(n_components=min(50, feat_s.shape[1]))
    z_s = pca50.fit_transform(feat_s)
    z_t = pca50.transform(feat_t)

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                perplexity=30, max_iter=1000)
    z = tsne.fit_transform(np.vstack([z_s, z_t]))
    ns = z_s.shape[0]
    z_s2, z_t2 = z[:ns], z[ns:]

    # t-SNE 图
    plt.figure(figsize=(9, 7))
    plt.scatter(z_s2[:, 0], z_s2[:, 1],
                s=25, c=y_s, cmap=cmap, vmin=0, vmax=9,
                alpha=0.45, marker='o', label="Source", edgecolors='none')
    plt.scatter(z_t2[:, 0], z_t2[:, 1],
                s=30, c=y_t, cmap=cmap, vmin=0, vmax=9,
                alpha=0.85, marker='^', label="Target", edgecolors='black', linewidths=0.3)

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

    plt.legend(handles=domain_handles + class_handles, frameon=True, ncol=5,
               fontsize=9, loc='best', title="Domains & Classes")

    plt.title(f"{title_prefix} | t-SNE", fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_tsne.png"), dpi=240)
    plt.close()

    # PCA 图
    p2 = PCA(n_components=2)
    zs = p2.fit_transform(feat_s)
    zt = p2.transform(feat_t)

    plt.figure(figsize=(9, 7))
    plt.scatter(zs[:, 0], zs[:, 1],
                s=25, c=y_s, cmap=cmap, vmin=0, vmax=9,
                alpha=0.45, marker='o', label="Source", edgecolors='none')
    plt.scatter(zt[:, 0], zt[:, 1],
                s=30, c=y_t, cmap=cmap, vmin=0, vmax=9,
                alpha=0.9, marker='^', label="Target", edgecolors='black', linewidths=0.3)
    # ✅ 同样修复 legend
    plt.legend(handles=domain_handles + class_handles, frameon=True, ncol=5,
               fontsize=9, loc='best', title="Domains & Classes")

    plt.title(f"{title_prefix} | PCA", fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_pca.png"), dpi=240)
    plt.close()


def plot_class_center_heatmap(feat_s, y_s, feat_t, y_t, num_classes, save_path, title="Center Dist (1-cos)"):
    """绘制源/目标类中心余弦距离热力图"""
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

    D = 1.0 - np.clip(Cs @ Ct.T, -1, 1)
    plt.figure(figsize=(6, 5))
    plt.imshow(D, interpolation='nearest', aspect='auto', cmap="YlGnBu")
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
                    pred_T: float = 1.0, conf_filter_quantile: float | None = None):
    """生成 t-SNE / PCA / 类中心热图 并返回统计指标"""
    ensure_dir(out_dir)
    feat_s, y_s, _  = collect_feats(src_model, src_loader, device, get_label=True,  pred_T=pred_T)
    feat_t, y_t, w  = collect_feats(tgt_model, tgt_loader, device, get_label=False, pred_T=pred_T)

    if conf_filter_quantile is not None:
        q = np.quantile(w, conf_filter_quantile)
        keep = (w >= q)
        feat_t, y_t, w = feat_t[keep], y_t[keep], w[keep]

    plot_tsne_pca(feat_s, y_s, feat_t, y_t,
                  os.path.join(out_dir, f"{epoch_tag}_vis.png"),
                  title_prefix=epoch_tag)

    diag = plot_class_center_heatmap(
        feat_s, y_s, feat_t, y_t, num_classes,
        os.path.join(out_dir, f"{epoch_tag}_center_heatmap.png"),
        title=f"{epoch_tag} | Center Dist (1-cos)"
    )

    p2 = PCA(n_components=2)
    js = js_divergence_2d(p2.fit_transform(feat_s), p2.transform(feat_t), bins=80)
    return {"center_diag": float(diag), "js2d": float(js)}
