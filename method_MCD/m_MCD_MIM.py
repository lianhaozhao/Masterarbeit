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
from models.get_no_label_dataloader import get_dataloaders
from models.MMD import infomax_loss_from_logits


# MCD 的分歧度量
def entropy(p, eps=1e-8):
    p = p.clamp(min=eps)
    return -(p * p.log()).sum()

def mutual_information(p1, p2, eps=1e-8):
    """
    估计 C1/C2 输出的互信息 I(Y1;Y2)  Mutual Information Minimization
    p1, p2: [B, C] softmax 概率
    """
    B, C = p1.size()
    p1_mean = p1.mean(0)          # [C]
    p2_mean = p2.mean(0)          # [C]
    joint = torch.einsum('bi,bj->ij', p1, p2) / B  # [C, C]

    H1 = entropy(p1_mean)
    H2 = entropy(p2_mean)
    H12 = entropy(joint.view(-1))

    return H1 + H2 - H12   # I(Y1;Y2)


#  评估（给出 C1/C2/Ensemble）
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
        list(model.feature_extractor.parameters()) + list(model.feature_reducer.parameters()) +
        list(model.c1.parameters()) + list(model.c2.parameters()),
        lr=lr_g, weight_decay=weight_decay
    )
    optim_C  = torch.optim.Adam(
        list(model.c1.parameters()) + list(model.c2.parameters()),
        lr=lr_c, weight_decay=weight_decay
    )
    optim_G  = torch.optim.Adam(
        list(model.feature_extractor.parameters()) + list(model.feature_reducer.parameters()),
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
            model.feature_extractor.train();model.feature_reducer.train();model.c1.train(); model.c2.train()
            optim_GC.zero_grad()
            l1s, l2s, _ = model(xs)
            loss_src = F.cross_entropy(l1s, ys) + F.cross_entropy(l2s, ys)
            lossA = loss_src
            lossA.backward(); optim_GC.step()
            sumA += lossA.item()
            countA +=1

            # ---- Step B: 固定 G，最大化目标域分歧（更新 C1/C2）----
            model.feature_extractor.eval()
            model.feature_reducer.eval()
            for p in model.feature_extractor.parameters(): p.requires_grad_(False)
            for p in model.feature_reducer.parameters(): p.requires_grad_(False)

            with torch.no_grad():
                ft = model.feature_extractor(xt)
                fs_b = model.feature_extractor(xs)
            model.c1.train()
            model.c2.train()
            for _ in range(nB):
                optim_C.zero_grad()
                l1t = model.c1(ft);
                l2t = model.c2(ft)
                p1, p2 = F.softmax(l1t, dim=1), F.softmax(l2t, dim=1)
                mi = mutual_information(p1, p2)

                # 同时维持源域能力
                ls1_b, ls2_b = model.c1(fs_b), model.c2(fs_b)
                loss_src_b = F.cross_entropy(ls1_b, ys) + F.cross_entropy(ls2_b, ys)

                # 目标：保持源域性能，同时最大化 C1/C2 在目标域的分歧 (最小化互信息)
                lossB = loss_src_b - lambda_dis * mi
                lossB.backward(); optim_C.step()
                sumB += lossB.item()
            countB += nB

            # ---- Step C: 固定 C1/C2，最小化目标域分歧（更新 G）----
            for p in model.feature_extractor.parameters(): p.requires_grad_(True)
            for p in model.feature_reducer.parameters(): p.requires_grad_(True)
            model.feature_extractor.train()
            model.feature_reducer.train()
            model.c1.eval()
            model.c2.eval()
            for p in model.c1.parameters(): p.requires_grad_(False)
            for p in model.c2.parameters(): p.requires_grad_(False)
            for _ in range(nC):
                optim_G.zero_grad()
                ft_c = model.feature_extractor(xt)
                lt1_c, lt2_c = model.c1(ft_c), model.c2(ft_c)
                p1_c, p2_c = F.softmax(lt1_c, dim=1), F.softmax(lt2_c, dim=1)

                mi_c = mutual_information(p1_c, p2_c)

                # InfoMax 正则（让预测更 confident 但不塌缩）
                loss_im_1, _, _ = infomax_loss_from_logits(lt1_c, T=1, marg_weight=1)
                loss_im_2, _, _ = infomax_loss_from_logits(lt2_c, T=1, marg_weight=1)
                loss_im = 0.5 * loss_im_1 + 0.5 * loss_im_2

                # 最小化互信息 + InfoMax
                lossC = lambda_dis * mi_c + loss_im
                lossC.backward()
                optim_G.step()
                sumC += lossC.item()
            countC += nC
            for p in model.c1.parameters(): p.requires_grad_(True)
            for p in model.c2.parameters(): p.requires_grad_(True)

            global_step += 1
        avg_lossA = sumA / max(1, countA)
        avg_lossB = sumB / max(1, countB)
        avg_lossC = sumC / max(1, countC)
        print(f"[Epoch {epoch:02d}] "
              f"lossA:{avg_lossA:.4f} lossB(last):{avg_lossB:.4f} lossC(last):{avg_lossC:.4f}")


        print("[INFO] Evaluating on target test set...")
        test_ds = PKLDataset(tgt_test)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

        ACC_A, ACC_B, ACC_AVG = MCD_evaluate(model, test_loader, device)
        print(f"ACC_A: {ACC_A:.04f}, ACC_B: {ACC_B:.04f}, ACC_AVG: {ACC_AVG:.04f}")


    # if best_state is not None:
    #     model.load_state_dict(best_state)

    return model

# ====== 入口 ======
if __name__ == "__main__":
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['baseline']
    bs = cfg['batch_size']; lr = cfg['learning_rate']; wd = cfg['weight_decay']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = 10

    # 数据路径
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
        lambda_dis=1.0, nB=1, nC=2
    )

    print("[INFO] Evaluating on target test set...")
    test_ds = PKLDataset(tgt_test)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    ACC_A, ACC_B, ACC_AVG= MCD_evaluate(model, test_loader, device)
    print(f"ACC_A: {ACC_A:.04f}, ACC_B: {ACC_B:.04f}, ACC_AVG: {ACC_AVG:.04f}")

