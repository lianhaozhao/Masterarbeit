import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import random
import yaml
from models.Flexible_DANN_MMD import Flexible_DANN
from PKLDataset import PKLDataset
from utils.general_train_and_test import general_test_model
from models.get_no_label_dataloader import get_dataloaders
from models.MMD import *
from collections import deque
import torch.nn.functional as F
import math


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_metric(curr, w_gap=0.5, w_mmd=0.1):
    """
    curr: {"gap": float, "mmd": float}
    - gap 越小越好 →   gap
    - mmd 越小越好 → 减去它们
    """
    return 0.5 - curr["gap"] * w_gap  - curr["mmd"] * w_mmd

def dann_lambda(epoch, num_epochs):
    """
    常用的 DANN λ 调度：从 0 平滑升到 0.6
    你也可以把 -10 调轻/重来改变上升速度
    """
    p = epoch / max(1, num_epochs - 1)
    return (2. / (1. + np.exp(-10 * p)) - 1.) * 0.5

def mmd_lambda(epoch, num_epochs, max_lambda=5e-2):
    # 0 → max_lambda，S 型上升
    p = epoch / max(1, num_epochs - 1)         # p ∈ [0,1]
    s = 1.0 / (1.0 + torch.exp(torch.tensor(-10.0*(p - 0.5))))  # ∈ (0,1)
    return float(max_lambda * s)

def train_dann_with_mmd(model, source_loader, target_loader,
                        optimizer, criterion_cls, criterion_domain,
                        device, num_epochs=20,
                        lambda_mmd_max=1e-1,           # MMD 最大权重
                        use_mk=True,                   # 多核 MMD
                        scheduler=None,
                        score_weights=(0.5, 0.1), # (w_gap, w_mmd)
                        warmup_best_start=10            # 多少个 epoch 后才开始考虑“最佳”
                        ):


    best_score = -float("inf")
    best_model_state = None
    patience = 0


    mmd_hist = deque(maxlen=4)  # 用最近5个epoch判断是否进入平台期
    gap_hist = deque(maxlen=4)

    mmd_fn = (lambda x, y: mmd_mk_biased(x, y, gammas=(0.5,1,2,4,8))) if use_mk \
             else (lambda x, y: mmd_rbf_biased_with_gamma(x, y, gamma=None))

    for epoch in range(num_epochs):
        cls_loss_sum, dom_loss_sum, mmd_loss_sum, total_loss_sum = 0.0, 0.0, 0.0, 0.0
        total_cls_samples, total_dom_samples = 0, 0
        dom_correct, dom_total = 0, 0
        model.train()


        for (src_x, src_y), tgt_x in zip(source_loader, target_loader):
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            lambda_dann_now = dann_lambda(epoch, num_epochs)
            cls_out_src, dom_out_src, feat_src = model(src_x, lambda_=lambda_dann_now)
            _, dom_out_tgt, feat_tgt = model(tgt_x, lambda_=lambda_dann_now)

            # 1) 分类损失（仅源域）
            loss_cls = criterion_cls(cls_out_src, src_y)

            # 2) 域分类损失（DANN）
            dom_label_src = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            dom_label_tgt = torch.ones(tgt_x.size(0),  dtype=torch.long, device=device)
            bs_src, bs_tgt = src_x.size(0), tgt_x.size(0)
            loss_dom_src = criterion_domain(dom_out_src, dom_label_src)
            loss_dom_tgt = criterion_domain(dom_out_tgt, dom_label_tgt)

            # 样本数加权的“单个域损失均值”
            loss_dom = (loss_dom_src * bs_src + loss_dom_tgt * bs_tgt) / (bs_src + bs_tgt)

            # 3) RBF‑MMD（特征对齐）
            # 先做 L2 归一化，提升稳定性
            feat_src_n = F.normalize(feat_src, dim=1)
            feat_tgt_n = F.normalize(feat_tgt, dim=1)
            loss_mmd = mmd_fn(feat_src_n, feat_tgt_n)

            # 4) 组合总损失
            #    - DANN 的 lambda 可继续用你已有的动态 dann_lambda
            #    - MMD 的权重做 warm‑up（避免一开始就把决策结构抹平）
            lambda_mmd_now  = float(mmd_lambda(epoch, num_epochs, max_lambda=lambda_mmd_max))

            loss = loss_cls + loss_dom + lambda_mmd_now * loss_mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录指标
            cls_loss_sum  += loss_cls.item() * src_x.size(0)
            dom_loss_sum  += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            mmd_loss_sum  += loss_mmd.item() * (src_x.size(0) + tgt_x.size(0))
            total_loss_sum += loss.item() * (src_x.size(0) + tgt_x.size(0))

            total_cls_samples += src_x.size(0)
            total_dom_samples += (src_x.size(0) + tgt_x.size(0))

            # 域分类准确率
            dom_preds_src = torch.argmax(dom_out_src, dim=1)
            dom_preds_tgt = torch.argmax(dom_out_tgt, dim=1)
            dom_correct += (dom_preds_src == dom_label_src).sum().item()
            dom_correct += (dom_preds_tgt == dom_label_tgt).sum().item()
            dom_total   += dom_label_src.size(0) + dom_label_tgt.size(0)

        # ——Epoch 级日志——
        avg_cls_loss  = cls_loss_sum  / max(1, total_cls_samples)
        avg_dom_loss  = dom_loss_sum  / max(1, total_dom_samples)
        avg_mmd_loss  = mmd_loss_sum  / max(1, total_dom_samples)
        avg_total_loss= total_loss_sum/ max(1, total_dom_samples)
        dom_acc = dom_correct / max(1, dom_total)
        gap = abs(dom_acc - 0.5)

        if scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch + 1}] Total loss: {avg_total_loss:.4f} | "
              f"Cls loss: {avg_cls_loss:.4f} | Dom avg loss: {avg_dom_loss:.4f} | "
              f"MMD avg loss: {avg_mmd_loss:.4f} | DomAcc: {dom_acc:.4f} | "
              f"λ_dann: {lambda_dann_now:.4f} | λ_mmd: {lambda_mmd_now:.4f}")

        mmd_hist.append(avg_mmd_loss)
        gap_hist.append(gap)

        # 选优
        curr = {"gap": gap, "mmd": avg_mmd_loss}
        w_gap, w_mmd = score_weights
        curr_score = score_metric(curr, w_gap=w_gap, w_mmd=w_mmd)
        improved = False
        if epoch >= warmup_best_start and (curr_score > best_score + 1e-6):
            best_score = curr_score
            best_model_state = copy.deepcopy(model.state_dict())
            improved = True

        # 早停判据
        MIN_EPOCH = 10
        PATIENCE = 4
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
        target_test_path = '../datasets/target/test/HC_T185_RP.txt'
        test_dataset = PKLDataset(target_test_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        general_test_model(model, criterion_cls, test_loader, device)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    return model


if __name__ == '__main__':
    set_seed(seed=44)
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
    target_path = '../datasets/target/train/HC_T185_RP.txt'
    target_test_path = '../datasets/target/test/HC_T185_RP.txt'
    out_path = 'model'
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Flexible_DANN(num_layers=num_layers,
                          start_channels=start_channels,
                          kernel_size=kernel_size,
                          cnn_act='leakrelu',
                          num_classes=10,
                          lambda_=1).to(device)

    source_loader, target_loader = get_dataloaders(source_path, target_path, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.1
    )
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    print("[INFO] Starting standard DANN training (no pseudo labels)...")
    model = train_dann_with_mmd(model, source_loader, target_loader,
                                optimizer, criterion_cls, criterion_domain,
                                device,
                                num_epochs=num_epochs,
                                lambda_mmd_max=1e-1,
                                use_mk=True,
                                scheduler=scheduler,
                                score_weights=(0.5, 0.1),  # 可按任务调整权重
                                warmup_best_start=3)

    print("[INFO] Evaluating on target test set...")
    test_dataset = PKLDataset(target_test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    general_test_model(model, criterion_cls, test_loader, device)
