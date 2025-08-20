import torch
import torch.nn.functional as F



# 距离与核函数基础工具


def pdist_squared(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    """
    计算两组样本的欧氏距离平方矩阵 ||x_i - y_j||^2

    参数：
        x: (N, D)
        y: (M, D) 或 None（表示与 x 自身配对）
    返回：
        dist2: (N, M)

    说明：
        使用 (x*x).sum(dim=1) 与 (y*y).sum(dim=1) 的展开式，数值更稳定。
        clamp_min_(0.0) 避免浮点误差导致的极小负数。
    """
    y = x if y is None else y
    x_norm = (x * x).sum(dim=1, keepdim=True)        # (N,1)
    y_norm = (y * y).sum(dim=1, keepdim=True).T      # (1,M)
    dist2 = x_norm + y_norm - 2.0 * (x @ y.T)        # (N,M)
    return dist2.clamp_min_(0.0)


@torch.no_grad()
def _median_gamma_concat(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    使用“median trick”在 x 与 y 的合并集合上估计单一带宽 γ，并返回 γ=1/(2*median_dist2)。

    步骤：
        1) 拼接 z=[x;y]，计算 z 的成对距离平方矩阵；
        2) 仅取上三角（去对角），并过滤掉 0；
        3) 取中位数 med2，返回 0.5/med2。

    注意：
        - 统一的 γ 会被 Kxx/Kyy/Kxy 共享，避免它们使用不同带宽导致数值不一致。
        - 若样本过少导致无有效距离，返回兜底 γ=1.0。
    """
    z = torch.cat([x, y], dim=0)
    d2 = pdist_squared(z, z)
    # 上三角去对角
    mask = torch.ones_like(d2, dtype=torch.bool)
    vals = d2[torch.triu(mask, diagonal=1)]
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return 1.0
    med2 = float(torch.clamp(vals.median(), min=eps))
    return 0.5 / med2


def rbf_kernel_with_gamma(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    RBF 核：K(x,y)=exp(-γ||x-y||^2)

    参数：
        x: (N,D)
        y: (M,D)
        gamma: 带宽参数 γ
    返回：
        K: (N,M)
    """
    d2 = pdist_squared(x, y)
    return torch.exp(-gamma * d2)



# 单核 MMD（支持权重/去对角）


def mmd_rbf_biased_with_gamma(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float,
    exclude_diag: bool = False,
) -> torch.Tensor:
    """
    单核 RBF 的 MMD^2（biased 估计；可选去对角）。

    公式（biased 版本）：
        MMD^2 = E[Kxx] + E[Kyy] - 2 E[Kxy]
    其中：
        - 若 exclude_diag=False，则 E[Kxx],E[Kyy] 为矩阵均值（含对角）；
        - 若 exclude_diag=True，则去掉对角后做均值（更接近 unbiased）。

    参数：
        x, y: (N,D), (M,D)
        gamma: 单一带宽 γ（建议使用 _median_gamma_concat 统一估的 γ）
        exclude_diag: 是否去对角自相似项
    返回：
        标量张量（MMD^2）
    """
    if gamma is None:
        gamma = _median_gamma_concat(x, y)
    Kxx = rbf_kernel_with_gamma(x, x, gamma)
    Kyy = rbf_kernel_with_gamma(y, y, gamma)
    Kxy = rbf_kernel_with_gamma(x, y, gamma)

    if exclude_diag:
        n, m = x.size(0), y.size(0)
        Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1) + 1e-12)
        Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1) + 1e-12)
    else:
        Kxx = Kxx.mean()
        Kyy = Kyy.mean()

    Kxy = Kxy.mean()
    return (Kxx + Kyy - 2.0 * Kxy).clamp_min(0.0)


def mmd_rbf_weighted_biased(
    x: torch.Tensor,
    y: torch.Tensor,
    w_x: torch.Tensor | None = None,
    w_y: torch.Tensor | None = None,
    gamma: float | None = None,
    exclude_diag: bool = False,
) -> torch.Tensor:
    """
    带样本权重的单核 RBF MMD^2（biased 估计；可选去对角）。

    参数：
        x, y: (N,D), (M,D)
        w_x, w_y: (N,), (M,) 原始非负权重；内部会做归一化使 sum=1
        gamma: 若为 None，会用 _median_gamma_concat(x,y) 估计统一 γ
        exclude_diag: 是否去掉对角项（Exx/Eyy 中的 self-similarity）

    返回：
        标量张量（MMD^2）

    权重版期望形式：
        Exx = Σ_i Σ_j (wx_i wx_j) Kxx_ij
        Eyy = Σ_i Σ_j (wy_i wy_j) Kyy_ij
        Exy = Σ_i Σ_j (wx_i wy_j) Kxy_ij
    """
    n, m = x.size(0), y.size(0)
    device = x.device

    if w_x is None:
        w_x = torch.ones(n, device=device)
    if w_y is None:
        w_y = torch.ones(m, device=device)

    wx = w_x.clamp_min(0).float()
    wy = w_y.clamp_min(0).float()
    wx = wx / (wx.sum() + 1e-12)
    wy = wy / (wy.sum() + 1e-12)

    if gamma is None:
        gamma = _median_gamma_concat(x, y)

    Kxx = rbf_kernel_with_gamma(x, x, gamma)
    Kyy = rbf_kernel_with_gamma(y, y, gamma)
    Kxy = rbf_kernel_with_gamma(x, y, gamma)

    if exclude_diag:
        Exx = (wx[:, None] * wx[None, :] * Kxx).sum() - (wx * wx * Kxx.diag()).sum()
        Eyy = (wy[:, None] * wy[None, :] * Kyy).sum() - (wy * wy * Kyy.diag()).sum()
    else:
        Exx = (wx[:, None] * wx[None, :] * Kxx).sum()
        Eyy = (wy[:, None] * wy[None, :] * Kyy).sum()

    Exy = (wx[:, None] * wy[None, :] * Kxy).sum()
    return (Exx + Eyy - 2.0 * Exy).clamp_min(0.0)



# 多核 MMD（保留，供通用使用）


def _mk_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    gammas: tuple[float, ...] = (0.5, 1, 2, 4, 8),
    base_gamma: float | None = None,
) -> torch.Tensor:
    """
    多核 RBF 的核矩阵，返回 **均值后** 的核（注意：做的是算术平均，不是求和）。

    参数：
        x, y: (N,D), (M,D)
        gammas: 核带宽倍数集；若 base_gamma 不为 None，则实际带宽为 base_gamma*γ_i
        base_gamma: 相对带宽基准；通常由 _median_gamma_concat 估计
    返回：
        K: (N,M) = (1/L) * Σ_i exp(-(base_gamma*γ_i)*||x-y||^2)

    说明：
        对不同 γ 的核做**算术平均**，可保持尺度与单核可比，避免无意扩大 L 倍。
    """
    d2 = torch.cdist(x, y, p=2.0) ** 2
    if base_gamma is not None:
        gamma_list = [base_gamma * g for g in gammas]
    else:
        gamma_list = list(gammas)
    K = sum(torch.exp(-g * d2) for g in gamma_list) / len(gamma_list)
    return K


def mmd_mk_biased(
    x: torch.Tensor,
    y: torch.Tensor,
    gammas: tuple[float, ...] = (0.5, 1, 2, 4, 8),
    use_relative: bool = True,
) -> torch.Tensor:
    """
    多核 RBF 的 MMD^2（biased 估计；不带样本权重）。

    参数：
        x, y: (N,D), (M,D)
        gammas: 多核带宽倍数
        use_relative: True 时使用 base_gamma（相对带宽）

    返回：
        标量张量（MMD^2）
    """
    base_gamma = _median_gamma_concat(x, y) if use_relative else None
    Kxx = _mk_rbf(x, x, gammas, base_gamma)
    Kyy = _mk_rbf(y, y, gammas, base_gamma)
    Kxy = _mk_rbf(x, y, gammas, base_gamma)
    return (Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()).clamp_min(0.0)


def mmd_mk_weighted_biased(
    x: torch.Tensor,
    y: torch.Tensor,
    w_x: torch.Tensor | None = None,
    w_y: torch.Tensor | None = None,
    gammas: tuple[float, ...] = (0.5, 1, 2, 4, 8),
    use_relative: bool = True,
) -> torch.Tensor:
    """
    **带样本权重**的多核 RBF MMD^2（biased 估计）。

    参数：
        x, y: (N,D), (M,D)
        w_x, w_y: (N,), (M,) 原始非负权重；内部会做归一化
        gammas: 多核的相对/绝对带宽列表
        use_relative: True 时先估计 base_gamma，并用 base_gamma*γ_i 作为实际带宽

    返回：
        标量张量（MMD^2）
    """
    n, m = x.size(0), y.size(0)
    device = x.device

    if w_x is None:
        w_x = torch.ones(n, device=device)
    if w_y is None:
        w_y = torch.ones(m, device=device)

    wx = w_x.clamp_min(0).float()
    wy = w_y.clamp_min(0).float()
    wx = wx / (wx.sum() + 1e-12)
    wy = wy / (wy.sum() + 1e-12)

    base_gamma = _median_gamma_concat(x, y) if use_relative else None

    Kxx = _mk_rbf(x, x, gammas, base_gamma)
    Kyy = _mk_rbf(y, y, gammas, base_gamma)
    Kxy = _mk_rbf(x, y, gammas, base_gamma)

    Exx = (wx[:, None] * wx[None, :] * Kxx).sum()
    Eyy = (wy[:, None] * wy[None, :] * Kyy).sum()
    Exy = (wx[:, None] * wy[None, :] * Kxy).sum()
    return (Exx + Eyy - 2.0 * Exy).clamp_min(0.0)



# 类条件 MMD（LMMD）—— 固定为“单核 + 样本级权重”


def classwise_mmd_biased(
    feat_src: torch.Tensor,
    y_src: torch.Tensor,
    feat_tgt: torch.Tensor,
    y_tgt: torch.Tensor,
    num_classes: int = 10,
    min_count_per_class: int = 2,
    weights_tgt: torch.Tensor | None = None,   # 仅目标域样本级权重（如伪标签置信度）
    gamma_conf: float | None = 2.0,            # 对目标权重做幂放大：w <- w^gamma_conf
    exclude_diag: bool = False,
) -> torch.Tensor:
    """
    类条件 MMD（LMMD），仅使用单核 RBF，并支持目标域样本级权重（如伪标签置信度）。

    用法建议：
        - y_tgt 为伪标签；weights_tgt 为伪标签置信度；
        - 设置 gamma_conf >= 1（如 1.5~2.0）可放大高置信度样本影响；
        - 类内带宽 γ 用 _median_gamma_concat(fs, ft) 统一估计；

    参数：
        feat_src, feat_tgt: (Ns,D), (Nt,D)
        y_src, y_tgt: (Ns,), (Nt,) 的整型类别 id（目标域常为伪标签）
        num_classes: 类别总数
        min_count_per_class: 每类最少样本阈值（源/目标域均需满足）
        weights_tgt: (Nt,) 目标域样本级原始非负权重（如置信度，未归一化）
        gamma_conf: 对 weights_tgt 的幂缩放（conf^γ）；None 或 1.0 表示不缩放
        exclude_diag: 是否在类内计算时去掉对角（仅影响单核 Exx/Eyy）

    类级权重（本版本仅用目标权重）：
        - 若提供了 weights_tgt，则 w_cls = min(ns, sum(weights_tgt_该类))；
        - 否则 w_cls = min(ns, nt)；
        - 最终对各类的 MMD^2 以 w_cls 做加权平均（对 w_cls 总和归一化）。

    返回：
        标量张量（LMMD^2）
    """
    total = feat_src.new_tensor(0.0)
    wsum = 0.0

    for c in range(num_classes):
        idx_s = (y_src == c)
        idx_t = (y_tgt == c)
        ns, nt = int(idx_s.sum().item()), int(idx_t.sum().item())
        if ns < min_count_per_class or nt < min_count_per_class:
            continue

        fs, ft = feat_src[idx_s], feat_tgt[idx_t]

        # 目标域原始（未归一化）样本权重
        wt_raw = weights_tgt[idx_t] if (weights_tgt is not None) else None
        if wt_raw is not None:
            wt_raw = wt_raw.clamp_min(0)
            if gamma_conf is not None and gamma_conf != 1.0:
                wt_raw = wt_raw.pow(gamma_conf)

        # 类内统一带宽（单核）
        gamma = _median_gamma_concat(fs, ft)

        # 单核 + 权重（源域等权，目标域用 wt_raw；函数内部会做归一化）
        mmd_c = mmd_rbf_weighted_biased(
            fs, ft,
            w_x=None,          # 源域等权
            w_y=wt_raw,        # 目标域样本级权重
            gamma=gamma,
            exclude_diag=exclude_diag
        )

        # 类级权重（交集思路）：避免目标域权重把源域极少样本的类放大过头
        if wt_raw is not None:
            wt_sum = float(wt_raw.sum().item())
            w_cls = min(float(ns), wt_sum)
        else:
            w_cls = float(min(ns, nt))

        if w_cls <= 0:
            continue

        total = total + mmd_c * w_cls
        wsum += w_cls

    return total / wsum if wsum > 0 else total




