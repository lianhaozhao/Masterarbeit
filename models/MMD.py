import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

#  Weighted Class Conditional MMD (Multi-core RBF)
def _pairwise_sq_dists(a, b):
    # a: [m,d], b: [n,d]
    a2 = (a*a).sum(dim=1, keepdim=True)       # [m,1]
    b2 = (b*b).sum(dim=1, keepdim=True).t()   # [1,n]
    return a2 + b2 - 2 * (a @ b.t())

def _mk_kernel(a, b, gammas):
    d2 = _pairwise_sq_dists(a, b).clamp_min(0)
    g = torch.as_tensor(gammas, dtype=d2.dtype, device=d2.device).view(-1, 1, 1)  # [G,1,1]
    K = torch.exp(-g * d2.unsqueeze(0))  # [G, m, n]
    return K.mean(dim=0)                 # [m, n]

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
    """
        Compute class-wise, biased MMD^2 between source and target features with
        importance weights on target samples.

        For each class c, this function:
          1. Selects source features feat_src[y_src == c] and target features
             feat_tgt[y_tgt == c]
          2. Uses the corresponding target weights w_tgt[y_tgt == c]
          3. Computes a (biased) weighted MMD^2 between the two sets via
             `mmd2_weighted`
          4. Weights the class-specific MMD^2 by w = min(n_s, n_t), where n_s and
             n_t are the sample counts of that class in source and target
          5. Averages over all classes that satisfy the minimum sample requirement

        Classes with too few samples in either domain (n_s < min_count_per_class
        or n_t < min_count_per_class) are ignored.

        Args
        ----
        feat_src : Tensor, shape [N_s, D]
            Source-domain features.
        y_src : LongTensor, shape [N_s]
            Source-domain class labels in [0, num_classes - 1].
        feat_tgt : Tensor, shape [N_t, D]
            Target-domain features.
        y_tgt : LongTensor, shape [N_t]
            Target-domain class pseudo labels in [0, num_classes - 1].
        w_tgt : Tensor, shape [N_t]
            Importance weights for target samples. Only entries corresponding
            to y_tgt == c are used for class c.
        num_classes : int
            Number of classes.
        gammas : tuple of float, default (0.5, 1, 2, 4, 8)
            Bandwidth parameters for the RBF kernels inside `mmd2_weighted`.
            Multiple gammas correspond to a mixture of RBF kernels.
        min_count_per_class : int, default 2
            Minimum number of samples required in BOTH source and target for a
            class to be included in the MMD computation.

        Returns
        -------
        mmd_classwise : Tensor (scalar)
            Class-wise, weighted average of biased MMD^2 over all eligible classes.
            If no class satisfies the minimum sample requirement, returns 0.0
            (as a tensor on the same device/dtype as feat_src).
        """
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
    Compute an unsupervised regularization term from classification logits
    based on the “information maximization” (InfoMax) principle:
    I(z; ŷ) = H(ŷ) - H(ŷ|z).
    During training we minimize:
        L = H(ŷ|z) - w * H(ŷ)
    where H(ŷ|z) is the conditional entropy (encourages confident per-sample
    predictions) and H(ŷ) is the marginal entropy (encourages balanced usage
    of classes and prevents collapse).

    Parameters
    ----------
    logits : Tensor
        Unnormalized scores from the classification head, of shape [B, C].
    T : float, default 1.0
        Softmax temperature. T > 1 makes the distribution “softer”
        (lower confidence), T < 1 makes it “sharper” (higher confidence).
        This affects both the conditional and marginal entropy values.
    marg_weight : float, default 1.0
        Weight w for the marginal entropy term. Larger values more strongly
        penalize solutions that collapse to a single class.
        A typical range is 0.5 ~ 2.0, to be tuned based on validation curves.

    Returns
    -------
    loss : Tensor (scalar, requires_grad=True)
        The objective to minimize: H(ŷ|z) - w * H(ŷ), used for backpropagation.
    h_cond_detached : Tensor (scalar, no grad)
        Empirical estimate of the conditional entropy H(ŷ|z),
        i.e., E_x[-∑_c p(c|x) log p(c|x)]. For logging/monitoring only.
    h_marg_detached : Tensor (scalar, no grad)
        Empirical estimate of the marginal entropy H(ŷ), where the marginal
        distribution of ŷ is p̄ = E_x[p(·|x)].
        Formally: H(ŷ) = -∑_c p̄_c log p̄_c. For logging/monitoring only.
"""

    # I(z;ŷ) = H(ŷ) - H(ŷ|z); minimiere  H(ŷ|z) - w * H(ŷ)
    p = F.softmax(logits / T, dim=1)  # [B, C], Klassenwahrscheinlichkeiten pro Beispiel
    h_cond = entropy_mean(p)  # Skalar, Schätzung der bedingten Entropie
    h_marg = entropy_marginal(p)  # Schätzung der marginalen Entropie
    return h_cond - marg_weight * h_marg, h_cond.detach(), h_marg.detach()

def infomax_loss_from_logits_2(
    logits,
    features=None,
    T=1.0,
    marg_weight=1.0,
    feat_weight=0.05,
    dynamic_balance=False,
    epoch=None,
    num_epochs=None,
):
    p = F.softmax(logits / T, dim=1)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-8)

    h_cond = entropy_mean(p)
    h_marg = entropy_marginal(p)

    feat_div = 0.0
    if features is not None:
        f_norm = F.normalize(features, dim=1)
        feat_div = f_norm.std(dim=0).mean()

    if dynamic_balance and (epoch is not None and num_epochs is not None):
        # Sigmoid 调度：平滑增长
        progress = 1 / (1 + np.exp(-10 * (epoch / num_epochs - 0.5)))
        feat_weight = feat_weight * progress
        marg_weight = marg_weight * (1.0 + 0.5 * progress)

    loss = h_cond - marg_weight * h_marg - feat_weight * feat_div
    return loss, h_cond.detach(), h_marg.detach()


class MultiPrototypes(nn.Module):
    def __init__(self, num_classes, feat_dim, num_protos=3, momentum=0.95):
        super().__init__()
        self.num_classes = num_classes
        self.num_protos = num_protos
        self.m = float(momentum)
        # [C, K, d]
        self.register_buffer('proto', torch.zeros(num_classes, num_protos, feat_dim))

    @torch.no_grad()
    def update(self, feats, labels, weights=None):
        if feats.numel() == 0:
            return
        device = feats.device  # Maintain consistency with input features
        for c in labels.unique():
            mask = (labels == c)
            if mask.sum() == 0: continue
            vec = feats[mask].mean(dim=0)

            proto_c = self.proto[c].to(device)

            sims = F.cosine_similarity(proto_c, vec.unsqueeze(0), dim=1)  # [K]
            k = sims.argmax().item()

            self.proto[c, k] = (
                    self.m * self.proto[c, k].to(device) + (1 - self.m) * vec
            )

    def supcon_logits(self, feats, tau=0.1, agg="max"):
        """
        feats: [N, d]
        return: [N, C]
        """
        f = F.normalize(feats, dim=1, eps=1e-8)             # [N, d]
        p = F.normalize(self.proto.to(feats.device), dim=2, eps=1e-8)      # [C, K, d]

        # [N, d] @ [C*K, d]^T -> [N, C*K]
        logits_all = f @ p.reshape(-1, p.size(-1)).t()
        logits_all = logits_all / float(tau)

        # reshape -> [N, C, K]
        logits_all = logits_all.view(feats.size(0), self.num_classes, self.num_protos)

        # aggregation
        if agg == "max":
            logits = logits_all.max(dim=2).values
        elif agg == "mean":
            logits = logits_all.mean(dim=2)
        elif agg == "lse":  # log-sum-exp
            logits = torch.logsumexp(logits_all, dim=2)
        else:
            raise ValueError(f"Unknown agg={agg}")
        return logits
