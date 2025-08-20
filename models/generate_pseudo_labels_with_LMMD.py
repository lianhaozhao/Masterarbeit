import torch
import torch.nn.functional as F


@torch.no_grad()
def ema_update(teacher, student, m: float = 0.999):
    """
    teacher = m * teacher + (1-m) * student
    同步所有参数与缓冲（包含 BN 的 running_mean/var 等）
    """
    t_state = teacher.state_dict()
    s_state = student.state_dict()
    for k in t_state.keys():
        # 有些 buffer 是整数（比如 num_batches_tracked），保持与 student 一致即可
        if t_state[k].dtype.is_floating_point:
            t_state[k].mul_(m).add_(s_state[k], alpha=1.0 - m)
        else:
            t_state[k].copy_(s_state[k])

def ema_momentum(step, m_max=0.999, m_min=0.9, warmup_steps=1000):
    if step >= warmup_steps:
        return m_max
    # 线性或余弦都行，这里用线性从 m_min → m_max
    r = step / max(1, warmup_steps)
    return m_min + (m_max - m_min) * r

@torch.no_grad()
def generate_pseudo_with_teacher(teacher, target_loader, device, threshold=0.9, return_conf=False):
    teacher.eval()
    xs, ys, confs = [], [], []
    for batch in target_loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        # grl=False，避免域对抗支路影响；eval() 已禁用 Dropout/BN 训练
        logits, _, _ = teacher(x, grl=False)
        prob = logits.softmax(dim=1)
        c, y = prob.max(dim=1)
        mask = c >= threshold     # 若想“软标签权重”，可以全收不做 mask
        xs.append(x[mask].cpu())
        ys.append(y[mask].cpu())
        confs.append(c[mask].cpu())
    if len(xs) == 0:
        if return_conf:
            return torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0)
        else:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
    px = torch.cat(xs); py = torch.cat(ys); pc = torch.cat(confs)
    return (px, py, pc) if return_conf else (px, py)

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


