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
from models.get_no_label_dataloader import get_target_loader
from models.MMD import *
from collections import deque
import torch.nn.functional as F



def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders(source_path, target_path, batch_size):
    source_dataset = PKLDataset(txt_path=source_path)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = get_target_loader(target_path, batch_size=batch_size, shuffle=True)
    return source_loader, target_loader

def dann_lambda(epoch, num_epochs):
    """
    常用的 DANN λ 调度：从 0 平滑升到 1
    你也可以把 -10 调轻/重来改变上升速度
    """
    p = epoch / float(num_epochs)
    return 2. / (1. + np.exp(-10 * p)) - 1.

def mmd_lambda(epoch, num_epochs, max_lambda=1e-1):
    # 0 → max_lambda，S 型上升
    p = epoch / max(1, num_epochs - 1)         # p ∈ [0,1]
    s = 1.0 / (1.0 + torch.exp(torch.tensor(-10.0*(p - 0.5))))  # ∈ (0,1)
    return float(max_lambda * s)

def train_dann_with_mmd(model, source_loader, target_loader,
                        optimizer, criterion_cls, criterion_domain,
                        device, num_epochs=20,
                        lambda_dann=0.1,           # 域分类器的权重
                        lambda_mmd_max=1e-1,       # MMD 的最大权重
                        use_mk=True,               # 是否用多核
                        scheduler=None):
    PATIENCE = 3
    MIN_EPOCH = 10

    best_gap = 0.5
    best_cls = float('inf')
    best_mmd = float('inf')
    best_model_state = None
    patience = 0

    MMD_THRESH = 4e-2  # MMD²足够小的阈值，按任务可调（0.02~0.05常见）
    MMD_PLATEAU_EPS = 5e-2  # 平台期判定的波动阈值
    mmd_hist = deque(maxlen=5)  # 用最近5个epoch判断是否进入平台期

    mmd_fn = (lambda x, y: mmd_mk_biased(x, y, gammas=(0.5,1,2,4,8))) if use_mk \
             else (lambda x, y: mmd_rbf_biased(x, y, gamma=None))

    for epoch in range(num_epochs):
        cls_loss_sum, dom_loss_sum, mmd_loss_sum, total_loss_sum = 0.0, 0.0, 0.0, 0.0
        total_cls_samples, total_dom_samples = 0, 0
        dom_correct, dom_total = 0, 0
        model.train()


        for (src_x, src_y), tgt_x in zip(source_loader, target_loader):
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            cls_out_src, dom_out_src, feat_src = model(src_x)
            _,            dom_out_tgt, feat_tgt = model(tgt_x)

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
            # 建议先做 L2 归一化，提升稳定性
            feat_src_n = F.normalize(feat_src, dim=1)
            feat_tgt_n = F.normalize(feat_tgt, dim=1)
            loss_mmd = mmd_fn(feat_src_n, feat_tgt_n)

            # 4) 组合总损失
            #    - DANN 的 lambda 可继续用你已有的动态 dann_lambda
            #    - MMD 的权重做 warm‑up（避免一开始就把决策结构抹平）
            lambda_dann_now = dann_lambda(epoch, num_epochs) if callable(lambda_dann) else lambda_dann
            lambda_mmd_now  = float(mmd_lambda(epoch, num_epochs, max_lambda=lambda_mmd_max))

            loss = loss_cls + lambda_dann_now * loss_dom + lambda_mmd_now * loss_mmd

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

        print(f"[Epoch {epoch + 1}] Total: {avg_total_loss:.4f} | "
              f"Cls: {avg_cls_loss:.4f} | Dom: {avg_dom_loss:.4f} | "
              f"MMD: {avg_mmd_loss:.4f} | DomAcc: {dom_acc:.4f} | "
              f"λ_dann: {lambda_dann_now:.4f} | λ_mmd: {lambda_mmd_now:.4f}")

        mmd_hist.append(avg_mmd_loss)
        mmd_plateau = (len(mmd_hist) == mmd_hist.maxlen) and (max(mmd_hist) - min(mmd_hist) < MMD_PLATEAU_EPS)

        # 触发条件
        cond_align = (gap < 0.05)
        cond_cls = (avg_cls_loss < 0.5)
        cond_mmd_small = (avg_mmd_loss < MMD_THRESH)
        cond_mmd_plateau = mmd_plateau

        # 是否有任何指标刷新“最好”
        improved = False
        if gap < best_gap - 1e-4:
            best_gap = gap
            best_model_state = copy.deepcopy(model.state_dict())
            improved = True

        # ——Early stopping：对齐 + 分类收敛 + （MMD小 和 MMD平台期）——
        if epoch > MIN_EPOCH and cond_align and cond_cls and (cond_mmd_small and cond_mmd_plateau):
            if not improved:
                patience += 1
            else:
                patience = 0
            print(f"[INFO] patience {patience} / {PATIENCE} | MMD_small={cond_mmd_small} plateau={cond_mmd_plateau}")
            if patience >= PATIENCE:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                print("[INFO] Early stopping: domain aligned, classifier converged, and MMD stabilized.")
                break
        else:

            patience = 0
        if epoch == (num_epochs-1):
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

    return model


if __name__ == '__main__':
    set_seed(seed=42)
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
                                device, num_epochs=30, lambda_dann=0.5, use_mk=True,scheduler=scheduler)

    print("[INFO] Evaluating on target test set...")
    test_dataset = PKLDataset(target_test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    general_test_model(model, criterion_cls, test_loader, device)
