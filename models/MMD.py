import torch
import torch.nn.functional as F

def pdist_squared(x, y=None):
    # x: [n,d], y: [m,d] or None
    y = x if y is None else y
    x_norm = (x**2).sum(dim=1, keepdim=True)     # [n,1]
    y_norm = (y**2).sum(dim=1, keepdim=True).T   # [1,m]
    dist2 = x_norm + y_norm - 2 * x @ y.T        # [n,m]
    return torch.clamp(dist2, min=0.0)

def rbf_kernel(x, y=None, gamma=None):
    # gamma = 1/(2*sigma^2)
    dist2 = pdist_squared(x, y)
    if gamma is None:
        # median trick：自适应带宽，减少调参
        with torch.no_grad():
            if y is None:
                triu = torch.triu(dist2, diagonal=1)
                med2 = triu[triu > 0].median()
            else:
                med2 = dist2.median()
            gamma = 0.5 / (med2 + 1e-12)
    K = torch.exp(-gamma * dist2)
    return K

def mmd_rbf_biased(x, y, gamma=None):
    """
    有偏估计（含对角），端到端训练更稳定
    x: [n,d], y: [m,d]
    """
    Kxx = rbf_kernel(x, x, gamma)
    Kyy = rbf_kernel(y, y, gamma)
    Kxy = rbf_kernel(x, y, gamma)
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

# ——可选：多核 RBF（更鲁棒），把上面 mmd_rbf_biased 换成这个即可——
def rbf_kernel_multi(x, y=None, gammas=(0.5, 1, 2, 4, 8)):
    dist2 = pdist_squared(x, y)
    Ks = [torch.exp(-g * dist2) for g in gammas]
    return sum(Ks) / len(Ks)

def mmd_mk_biased(x, y, gammas=(0.5, 1, 2, 4, 8)):
    Kxx = rbf_kernel_multi(x, x, gammas)
    Kyy = rbf_kernel_multi(y, y, gammas)
    Kxy = rbf_kernel_multi(x, y, gammas)
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
