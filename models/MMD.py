import torch
import torch.nn.functional as F


#  Weighted Class Conditional MMD (Multi-core RBF)
def _pairwise_sq_dists(a, b):
    # a: [m,d], b: [n,d]
    a2 = (a*a).sum(dim=1, keepdim=True)       # [m,1]
    b2 = (b*b).sum(dim=1, keepdim=True).t()   # [1,n]
    return a2 + b2 - 2 * (a @ b.t())

def _mk_kernel(a, b, gammas):
    d2 = _pairwise_sq_dists(a, b).clamp_min(0)
    k = 0.0
    M = max(1, len(gammas))
    for g in gammas:
        k = k + torch.exp(-float(g) * d2)
    return k / M

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
@torch.no_grad()
def suggest_mmd_gammas(x_src, x_tgt, scales=(0.25,0.5,1,2,4)):
    x = torch.cat([x_src.detach(), x_tgt.detach()], dim=0)

    xi, xj = x.unsqueeze(1), x.unsqueeze(0)
    d2 = (xi - xj).pow(2).sum(-1).flatten()

    m = d2.clamp_min(1e-12).median()
    g0 = (1.0 / (2.0 * m)).item()
    return [s * g0 for s in scales]

def mmd2_unconditional(feat_src, feat_tgt, gammas):
    return mmd2_weighted(feat_src, feat_tgt, w_a=None, w_b=None, gammas=gammas)

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
