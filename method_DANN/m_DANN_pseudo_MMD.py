import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import random
import yaml
from models.Flexible_DANN_pseudo_MMD import Flexible_DANN
from PKLDataset import PKLDataset
from utils.general_train_and_test import general_test_model
from models.get_no_label_dataloader import get_target_loader
from models.MMD import *
from collections import deque
import torch.nn.functional as F
import math
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_labels


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders(source_path, target_path, batch_size):
    source_dataset = PKLDataset(txt_path=source_path)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = get_target_loader(target_path, batch_size=batch_size, shuffle=True)  # :contentReference[oaicite:4]{index=4}
    return source_loader, target_loader

def score_metric(curr, w_gap=1.0, w_mmd=0.1):
    """
    curr: {"gap": float, "mmd": float}
    - gap 越小越好 →  gap
    - cls/mmd 越小越好 → 减去它们
    """
    return 0.5 - curr["gap"] * w_gap - curr["mmd"] * w_mmd

def dann_lambda(epoch, num_epochs, max_lambda=0.5):
    """
    DANN λ 从 epoch=0 就启用，S 型从 0 → max_lambda
    """
    p = epoch / max(1, num_epochs-1)   # 归一化到 [0,1]
    return (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) * max_lambda

def mmd_lambda(epoch, num_epochs, max_lambda=5e-2, start_epoch=4):
    if epoch < start_epoch: return 0.0
    p = (epoch-start_epoch)/max(1,(num_epochs-1-start_epoch))
    s = 1/(1+math.exp(-10*(p-0.5)))
    return float(max_lambda*s)


# ---------- DANN + LMMD（伪标签 + 置信过滤） ----------
def train_dann_with_clmmd(model, source_loader, target_loader,
                          optimizer, criterion_cls, criterion_domain,
                          device, num_epochs=20,
                          num_classes=10,
                          pseudo_thresh=0.95,  # 伪标签置信阈值
                          lambda_mmd_max=5e-2,  # LMMD 最大权重（warm-up）
                          score_weights=(0.8, 0.1),
                          gammas=1.5,
                          scheduler=None,
                          batch_size=16):
    warmup_best_start = 10
    global_step = 0
    best_score = -float("inf")
    best_model_state = None
    patience = 0

    mmd_hist = deque(maxlen=3)
    gap_hist = deque(maxlen=3)

    #  training
    for epoch in range(num_epochs):

        # A) 每个 epoch 先用当前模型，在整套 target 上生成“硬伪标签”（按阈值过滤）
        #    返回的是 CPU 张量；若为空则 DataLoader 为 None
        pl_loader = None
        pseudo_x = torch.empty(0)
        pseudo_y = torch.empty(0, dtype=torch.long)
        if epoch > 3:
            pseudo_x, pseudo_y, pseudo_w = generate_pseudo_labels(model, target_loader, device, threshold=pseudo_thresh)  #
            if pseudo_x.numel() > 0:
                pl_dataset = TensorDataset(pseudo_x, pseudo_y, pseudo_w)
                pl_loader = DataLoader(pl_dataset, batch_size=batch_size, shuffle=True)
            else:
                pl_loader = None
                print("no pseudo data")
        model.train()

        # 迭代步数：对齐三方（源、有标签伪目标、无标签目标）长度，轮转较短的一侧
        len_src = len(source_loader)
        len_tgt = len(target_loader)
        num_iters = max(len_src, len_tgt)

        it_src = iter(source_loader)
        it_tgt = iter(target_loader)
        it_pl  = iter(pl_loader) if pl_loader is not None else None

        # 累计日志量
        cls_loss_sum = dom_loss_sum = mmd_loss_sum = total_loss_sum = 0.0
        total_cls_samples = total_dom_samples = 0
        dom_correct_src = 0
        dom_total_src = 0
        dom_correct_tgt = 0
        dom_total_tgt = 0
        pseudo_kept = pseudo_x.size(0) if pseudo_x.numel() > 0 else 0
        tgt_total_epoch = len(target_loader.dataset)  # 粗略统计

        for _ in range(num_iters):
            # 取 batch（轮转）
            try: (src_x, src_y) = next(it_src)
            except StopIteration:
                it_src = iter(source_loader); (src_x, src_y) = next(it_src)

            try: tgt_x = next(it_tgt)
            except StopIteration:
                it_tgt = iter(target_loader); tgt_x = next(it_tgt)

            if isinstance(tgt_x, (tuple, list)):
                tgt_x = tgt_x[0]

            if it_pl is not None:
                try: (tgt_pl_x, tgt_pl_y, tgt_pl_w) = next(it_pl)
                except StopIteration:
                    it_pl = iter(pl_loader); (tgt_pl_x, tgt_pl_y, tgt_pl_w) = next(it_pl)
            else:
                tgt_pl_x = tgt_pl_y = tgt_pl_w =None

            # 送设备
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)
            if tgt_pl_x is not None:
                tgt_pl_x = tgt_pl_x.to(device)
                tgt_pl_y = tgt_pl_y.to(device)
                tgt_pl_w = tgt_pl_w.to(device)

            # 前向
            # lambda_dann_now = dann_lambda(epoch, num_epochs)
            lambda_dann_now = 0.5
            cls_out_src, dom_out_src, feat_src = model(src_x,lambda_=lambda_dann_now)
            _,           dom_out_tgt, feat_tgt = model(tgt_x,lambda_=lambda_dann_now)

            # 1) 分类损失（源域）
            loss_cls = criterion_cls(cls_out_src, src_y)

            # 2) 域对抗损失（源/目标无标签）
            dom_label_src = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            dom_label_tgt = torch.ones(tgt_x.size(0),  dtype=torch.long, device=device)
            bs_src, bs_tgt = src_x.size(0), tgt_x.size(0)
            loss_dom_src = criterion_domain(dom_out_src, dom_label_src)
            loss_dom_tgt = criterion_domain(dom_out_tgt, dom_label_tgt)
            loss_dom = (loss_dom_src * bs_src + loss_dom_tgt * bs_tgt) / (bs_src + bs_tgt)

            # 3) 类条件对齐（LMMD）：仅对“高置信伪标签”的目标样本做类内 MMD
            #    - 使用你已有的多核 MMD 实现；特征先 L2 归一化更稳
            feat_src_n = F.normalize(feat_src, dim=1)
            if tgt_pl_x is not None:
                # 为了 LMMD，需要目标端的“伪标签”对应的特征，单独前向取特征

                cls_out_pl, _, feat_tgt_pl = model(tgt_pl_x,lambda_=0)
                feat_tgt_pl_n = F.normalize(feat_tgt_pl, dim=1)

                loss_lmmd = classwise_mmd_biased(
                    feat_src_n, src_y,
                    feat_tgt_pl_n, tgt_pl_y,
                    num_classes=num_classes,
                    min_count_per_class=2,
                    weights_tgt=tgt_pl_w ,
                    gamma_conf=gammas,
                )

            else:
                loss_lmmd = feat_src_n.new_tensor(0.0)

            # 4) 组合损失（DANN 调度 + LMMD warm-up）
            # lambda_mmd_now  = float(mmd_lambda(epoch, num_epochs, max_lambda=lambda_mmd_max))
            lambda_mmd_now = 0.05
            loss = loss_cls +  loss_dom + lambda_mmd_now * loss_lmmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # ← 新位置（从 epoch 末尾移到这里）

            global_step += 1

            #  记录指标
            cls_loss_sum  += loss_cls.item() * src_x.size(0)
            dom_loss_sum  += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            mmd_loss_sum  += loss_lmmd.item() * src_x.size(0)
            total_loss_sum += loss.item() * (src_x.size(0) + tgt_x.size(0))
            total_cls_samples += src_x.size(0)
            total_dom_samples += (src_x.size(0) + tgt_x.size(0))
            # 域分类准确率
            dom_preds_src = torch.argmax(dom_out_src, dim=1)
            dom_preds_tgt = torch.argmax(dom_out_tgt, dim=1)
            dom_correct_src += (dom_preds_src == dom_label_src).sum().item()
            dom_total_src += dom_label_src.size(0)
            dom_correct_tgt += (dom_preds_tgt == dom_label_tgt).sum().item()
            dom_total_tgt += dom_label_tgt.size(0)

        # ——Epoch 级日志——
        avg_cls_loss   = cls_loss_sum   / max(1, total_cls_samples)
        avg_dom_loss   = dom_loss_sum   / max(1, total_dom_samples)
        avg_lmmd_loss  = mmd_loss_sum   / max(1, total_cls_samples)
        avg_total_loss = total_loss_sum / max(1, total_dom_samples)
        acc_src = dom_correct_src / max(1, dom_total_src)
        acc_tgt = dom_correct_tgt / max(1, dom_total_tgt)
        dom_acc = 0.5 * (acc_src + acc_tgt)
        gap = abs(dom_acc - 0.5)


        kept_ratio = (pseudo_kept / max(1, tgt_total_epoch)) if tgt_total_epoch else 0.0
        print(f"[Epoch {epoch + 1}] Total loss: {avg_total_loss:.4f} | "
              f"Cls loss: {avg_cls_loss:.4f} | Dom loss: {avg_dom_loss:.4f} | "
              f"LMMD loss: {avg_lmmd_loss:.4f} | DomAcc: {dom_acc:.4f} | "
              f"KeepPL: {pseudo_kept}/{tgt_total_epoch} ({kept_ratio:.2%}) | "
              f"λ_dann: {lambda_dann_now:.4f} | λ_mmd: {lambda_mmd_now:.4f}"
              )

        mmd_hist.append(avg_lmmd_loss)
        gap_hist.append(gap)

        # 选优
        curr = {"gap": gap, "mmd": avg_lmmd_loss}
        w_gap, w_mmd = score_weights
        curr_score = score_metric(curr, w_gap=w_gap, w_mmd=w_mmd)
        improved = False
        if epoch >= warmup_best_start and (curr_score > best_score + 1e-6):
            best_score = curr_score
            best_model_state = copy.deepcopy(model.state_dict())
            improved = True

        # 早停判据
        MIN_EPOCH = 10
        PATIENCE = 3
        GAP = 0.05
        EPS_REL = 0.05  # LMMD 相对变化 <5% 视为平台期

        # 1) 对齐“软门槛”：最近窗口平均 gap 很小
        gap_ok = (len(gap_hist) == gap_hist.maxlen) and (sum(gap_hist) / len(gap_hist) < GAP)

        # 2) LMMD 相对平台：最近窗口的首尾相差占比很小
        if len(mmd_hist) == mmd_hist.maxlen:
            lmmd_first, lmmd_last = mmd_hist[0], mmd_hist[-1]
            denom = max(1e-8, max(lmmd_first, lmmd_last))
            lmmd_plateau_rel = (abs(lmmd_last - lmmd_first) / denom) < EPS_REL
        else:
            lmmd_plateau_rel = False

        # 3) 主判据：score 没提升就累计耐心；同时要求“对齐达标 + LMMD 进入平台”
        if epoch >= MIN_EPOCH and gap_ok and lmmd_plateau_rel:
            patience = 0 if improved else (patience + 1)
            print(f"[INFO] patience {patience}/{PATIENCE} | gap_ok={gap_ok} | lmmd_plateau_rel={lmmd_plateau_rel}")
            if patience >= PATIENCE:
                print("[INFO] Early stopping by score patience with stable alignment/MMD.")
                model.load_state_dict(best_model_state)
                break
        else:
            patience = 0
        print("[INFO] Evaluating on target test set...")
        target_test_path = '../datasets/target/test/HC_T188_RP.txt'
        test_dataset = PKLDataset(target_test_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        general_test_model(model, criterion_cls, test_loader, device)

    # 结束时回载
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model
if __name__ == '__main__':
    # set_seed(seed=44)
    with open("../configs/default.yaml", 'r') as f:
        config = yaml.safe_load(f)['baseline']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_layers = config['num_layers']
    kernel_size = config['kernel_size']
    start_channels = config['start_channels']
    num_epochs = config['num_epochs']

    source_path = '../datasets/source/train/DC_T197_RP.txt'
    target_path = '../datasets/target/train/HC_T188_RP.txt'
    target_test_path = '../datasets/target/test/HC_T188_RP.txt'
    out_path = 'model'
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 10

    model = Flexible_DANN(num_layers=num_layers,
                          start_channels=start_channels,
                          kernel_size=kernel_size,
                          cnn_act='leakrelu',
                          num_classes=NUM_CLASSES,
                          lambda_=0.5).to(device)

    source_loader, target_loader = get_dataloaders(source_path, target_path, batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    steps_per_epoch = max(len(source_loader), len(target_loader))
    # 每 3 个 epoch 重启一次；想固定每3e都重启，用 T_mult=1
    T_0 = 3 * steps_per_epoch
    warmup_iters = int(0.05 * steps_per_epoch * num_epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    cosine_wr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1,
                                                                     eta_min=learning_rate * 0.1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine_wr], milestones=[warmup_iters]
    )
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    print("[INFO] Starting DANN + LMMD (pseudo labels + confidence filtering)...")
    model = train_dann_with_clmmd(
        model, source_loader, target_loader,
        optimizer, criterion_cls, criterion_domain,
        device,
        num_epochs=num_epochs,
        num_classes=NUM_CLASSES,
        pseudo_thresh=0.95,          # 可调：0.8~0.95 常见
        lambda_mmd_max=1e-1,
        gammas=1.5,
        scheduler=scheduler,
        batch_size=batch_size

    )

    print("[INFO] Evaluating on target test set...")
    test_dataset = PKLDataset(target_test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    general_test_model(model, criterion_cls, test_loader, device)
