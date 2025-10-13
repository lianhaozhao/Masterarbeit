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
