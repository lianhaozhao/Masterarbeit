import copy, math, random, os
import yaml
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.Flexible_CNN_MCD import Flexible_MCD
from PKLDataset import PKLDataset
from models.get_no_label_dataloader import get_target_loader
from utils.general_train_and_test import general_test_model


def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_dataloaders(source_path, target_path, batch_size):
    src_ds = PKLDataset(txt_path=source_path)
    tgt_loader = get_target_loader(target_path, batch_size=batch_size, shuffle=True)
    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True)
    return src_loader, tgt_loader

# MCD 的分歧度量
def discrepancy(logits1, logits2, reduction='mean'):
    p1, p2 = F.softmax(logits1, dim=1), F.softmax(logits2, dim=1)
    d = (p1 - p2).abs().sum(1)
    return d.mean() if reduction == 'mean' else d.sum() if reduction == 'sum' else d

# ====== 评估（给出 C1/C2/Ensemble） ======
@torch.no_grad()
def MCD_evaluate(model, loader, device):
    model.eval()
    n = 0; top1 = top2 = topE = 0
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0].to(device), batch[1].to(device).long()
        else:
            # 若没有标签，直接跳过
            continue
        l1, l2, _ = model(x)
        p1, p2 = F.softmax(l1, 1), F.softmax(l2, 1)
        pe = (p1 + p2) / 2
        n  += y.size(0)
        top1 += (l1.argmax(1) == y).sum().item()
        top2 += (l2.argmax(1) == y).sum().item()
        topE += (pe.argmax(1) == y).sum().item()
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return top1/n, top2/n, topE/n

# ====== MCD 三步训练 ======
def train_mcd(model, src_loader, tgt_loader, device,
              num_epochs=50, lr_g=2e-4, lr_c=2e-4, weight_decay=0.0,
              lambda_dis=1.0, nB=4, nC=4,
              ):

    model.to(device)
    model.train()

    # 优化器：GC / 仅C / 仅G
    optim_GC = torch.optim.Adam(
        list(model.feature_extractor.parameters()) +
        list(model.c1.parameters()) + list(model.c2.parameters()),
        lr=lr_g, weight_decay=weight_decay
    )
    optim_C  = torch.optim.Adam(
        list(model.c1.parameters()) + list(model.c2.parameters()),
        lr=lr_c, weight_decay=weight_decay
    )
    optim_G  = torch.optim.Adam(
        model.feature_extractor.parameters(),
        lr=lr_g, weight_decay=weight_decay
    )

    global_step = 0

    best_state = None

    for epoch in range(1, num_epochs+1):
        iters = max(len(src_loader), len(tgt_loader))
        it_src, it_tgt = iter(src_loader), iter(tgt_loader)

        sumA = sumB = sumC = 0.0
        countA = countB = countC = 0
        for it in range(iters):
            try: xs, ys = next(it_src)
            except StopIteration: it_src = iter(src_loader); xs, ys = next(it_src)
            try: xt = next(it_tgt)
            except StopIteration: it_tgt = iter(tgt_loader); xt = next(it_tgt)

            xs, ys, xt = xs.to(device), ys.to(device).long(), xt.to(device)

            # ---- Step A: 源域监督 (更新 G, C1, C2) ----
            model.feature_extractor.train();model.c1.train(); model.c2.train()
            optim_GC.zero_grad()
            l1s, l2s, _ = model(xs)
            loss_src = F.cross_entropy(l1s, ys) + F.cross_entropy(l2s, ys)
            lossA = loss_src
            lossA.backward(); optim_GC.step()
            sumA += lossA.item()
            countA +=1

            # ---- Step B: 固定 G，最大化目标域分歧（更新 C1/C2）----
            model.feature_extractor.eval()  # 避免 BN 统计量被更新
            model.c1.train()
            model.c2.train()
            for p in model.feature_extractor.parameters(): p.requires_grad_(False)
            for _ in range(nB):
                optim_C.zero_grad()
                ft = model.feature_extractor(xt).detach()      # 固定 G
                l1t = model.c1(ft); l2t = model.c2(ft)
                disc_t = discrepancy(l1t, l2t, 'mean')

                # 同时维持源域能力，避免崩坏（源域 CE）
                fs_b = model.feature_extractor(xs).detach()       # 通过图传播到 C1/C2
                ls1_b, ls2_b = model.c1(fs_b), model.c2(fs_b)
                loss_src_b = F.cross_entropy(ls1_b, ys) + F.cross_entropy(ls2_b, ys)

                lossB = loss_src_b - lambda_dis * disc_t   # 最小化该式 => 最大化分歧
                lossB.backward(); optim_C.step()
                sumB += lossB.item()
            countB += nB

            # ---- Step C: 固定 C1/C2，最小化目标域分歧（更新 G）----
            for p in model.feature_extractor.parameters(): p.requires_grad_(True)
            model.feature_extractor.train()
            model.c1.eval()
            model.c2.eval()
            for p in model.c1.parameters(): p.requires_grad_(False)
            for p in model.c2.parameters(): p.requires_grad_(False)
            for _ in range(nC):
                optim_G.zero_grad()
                ft_c = model.feature_extractor(xt)
                lt1_c, lt2_c = model.c1(ft_c), model.c2(ft_c)
                disc_c = discrepancy(lt1_c, lt2_c, 'mean')
                lossC = lambda_dis * disc_c
                lossC.backward(); optim_G.step()
                sumC += lossC.item()
            countC += nC
            for p in model.c1.parameters(): p.requires_grad_(True)
            for p in model.c2.parameters(): p.requires_grad_(True)

            global_step += 1
        avg_lossA = sumA / max(1, countA)
        avg_lossB = sumB / max(1, countB)
        avg_lossC = sumC / max(1, countC)
        print(f"[Epoch {epoch:03d} | Step {global_step:05d}] "
              f"lossA:{avg_lossA:.4f} lossB(last):{avg_lossB:.4f} lossC(last):{avg_lossC:.4f}")


        print("[INFO] Evaluating on target test set...")
        test_ds = PKLDataset(tgt_test)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

        ACC_A, ACC_B, ACC_AVG = MCD_evaluate(model, test_loader, device)
        print(f"ACC_A: {ACC_A}, ACC_B: {ACC_B}, ACC_AVG: {ACC_AVG}")


    if best_state is not None:
        model.load_state_dict(best_state)

    return model

# ====== 入口 ======
if __name__ == "__main__":
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['baseline']
    bs = cfg['batch_size']; lr = cfg['learning_rate']; wd = cfg['weight_decay']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = cfg['num_epochs']

    # 数据路径（与你原脚本保持一致）
    src_path = "../datasets/source/train/DC_T197_RP.txt"
    tgt_path = "../datasets/target/train/HC_T185_RP.txt"
    tgt_test = "../datasets/target/test/HC_T185_RP.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建数据
    src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)

    val_loader = DataLoader(PKLDataset(tgt_test), batch_size=bs, shuffle=False)

    # 构建模型
    model = Flexible_MCD(
        num_layers=num_layers, start_channels=sc, kernel_size=ksz, cnn_act='leakrelu',
        num_classes=10
    )

    # 训练（MCD）
    model = train_mcd(
        model, src_loader, tgt_loader, device,
        num_epochs=20, lr_g=lr, lr_c=lr, weight_decay=wd,
        lambda_dis=1.0, nB=4, nC=5
    )

    print("[INFO] Evaluating on target test set...")
    test_ds = PKLDataset(tgt_test)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    ACC_A, ACC_B, ACC_AVG= MCD_evaluate(model, test_loader, device)
    print(f"ACC_A: {ACC_A}, ACC_B: {ACC_B}, ACC_AVG: {ACC_AVG}")

