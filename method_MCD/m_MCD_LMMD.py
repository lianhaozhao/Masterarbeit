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
from models.get_no_label_dataloader import get_dataloaders,get_pseudo_dataloaders
from models.MMD import infomax_loss_from_logits
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_with_stats
from models.MMD import classwise_mmd_biased_weighted,suggest_mmd_gammas, infomax_loss_from_logits


# MCD 的分歧度量
def discrepancy(logits1, logits2, reduction='mean'):
    p1, p2 = F.softmax(logits1, dim=1), F.softmax(logits2, dim=1)
    d = (p1 - p2).abs().sum(1)
    return d.mean() if reduction == 'mean' else d.sum() if reduction == 'sum' else d

def mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=5):
    if epoch < start_epoch:
        return 0.0
    p = (epoch - start_epoch) / max(1, (num_epochs - 1 - start_epoch))
    return (2.0 / (1.0 + np.exp(-3 * p)) - 1.0) * max_lambda
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
              lmmd_start_epoch = 5,ps_loader = None,max_lambda = 0.35,
              num_epochs=50, lr_g=2e-4, lr_c=2e-4, weight_decay=0.0,
              lambda_dis=1.0, nB=4, nC=4,
              save_dir="/content/drive/MyDrive/MCD/",
              tag="T000_RUN0"
              ):

    model.to(device)
    model.train()
    os.makedirs(save_dir, exist_ok=True)

    # 优化器：GC / 仅C / 仅G
    optim_GC = torch.optim.AdamW(
        list(model.feature_extractor.parameters()) + list(model.feature_reducer.parameters()) +
        list(model.c1.parameters()) + list(model.c2.parameters()),
        lr=lr_g, weight_decay=weight_decay
    )
    optim_C  = torch.optim.AdamW(
        list(model.c1.parameters()) + list(model.c2.parameters()),
        lr=lr_c, weight_decay=weight_decay
    )
    optim_G  = torch.optim.AdamW(
        list(model.feature_extractor.parameters()) + list(model.feature_reducer.parameters()),
        lr=lr_g, weight_decay=weight_decay
    )

    global_step = 0

    best_state = None
    best_score = float("inf")

    for epoch in range(1, num_epochs+1):
        # 1) Pseudo-labeling
        pl_loader = None
        cached_gammas = None
        pseudo_x = torch.empty(0)
        pseudo_y = torch.empty(0, dtype=torch.long)
        pseudo_w = torch.empty(0)
        cov = margin_mean = 0.0
        if epoch >= lmmd_start_epoch:
            pseudo_x, pseudo_y, pseudo_w, stats = generate_pseudo_with_stats(
                model, ps_loader, device, threshold=0.95, T=2
            )
            kept, total = stats["kept"], stats["total"]
            cov, margin_mean = stats["coverage"], stats["margin_mean"]
            if kept > 0:
                pl_ds = TensorDataset(pseudo_x, pseudo_y, pseudo_w)
                pl_loader = DataLoader(pl_ds, batch_size=64, shuffle=True, drop_last=False)

        # 2) Gated LMMD weights
        lambda_lmmd_eff = mmd_lambda(epoch, num_epochs, max_lambda=max_lambda, start_epoch=lmmd_start_epoch)

        # 3) epoch training
        iters = max(len(src_loader), len(tgt_loader))
        it_src, it_tgt = iter(src_loader), iter(tgt_loader)
        it_pl = iter(pl_loader) if pl_loader is not None else None

        sumA = sumB = sumC = 0.0
        countA = countB = countC = im  = 0

        for it in range(iters):
            try: xs, ys = next(it_src)
            except StopIteration: it_src = iter(src_loader); xs, ys = next(it_src)
            try: xt = next(it_tgt)
            except StopIteration: it_tgt = iter(tgt_loader); xt = next(it_tgt)

            xs, ys, xt = xs.to(device), ys.to(device).long(), xt.to(device)
            if it_pl is not None:
                try:
                    xpl, ypl, wpl = next(it_pl)
                except StopIteration:
                    it_pl = iter(pl_loader)
                    xpl, ypl, wpl = next(it_pl)
                xpl, ypl, wpl = xpl.to(device), ypl.to(device), wpl.to(device)
            else:
                xpl = ypl = wpl = None

            # ---- Step A: 源域监督 (更新 G, C1, C2) ----
            model.feature_extractor.train();model.feature_reducer.train();model.c1.train(); model.c2.train()
            optim_GC.zero_grad()
            for _ in range(3):
                l1s, l2s, _ = model(xs)
                loss_src = F.cross_entropy(l1s, ys) + F.cross_entropy(l2s, ys)
                lossA = loss_src
                lossA.backward()
                optim_GC.step()
                sumA += lossA.item()
                countA += 1

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
                l1t = model.c1(ft); l2t = model.c2(ft)
                disc_t = discrepancy(l1t, l2t, 'mean')

                # 同时维持源域能力，避免崩坏（源域 CE）
                ls1_b, ls2_b = model.c1(fs_b), model.c2(fs_b)
                loss_src_b = F.cross_entropy(ls1_b, ys) + F.cross_entropy(ls2_b, ys)
                lossB = loss_src_b * 0.3 - lambda_dis * disc_t   # 最小化该式 => 最大化分歧
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
                _, h_cond_1, h_marg_1 = infomax_loss_from_logits(lt1_c, T=1, marg_weight=1)
                _, h_cond_2, h_marg_2 = infomax_loss_from_logits(lt2_c, T=1, marg_weight=1)
                im += ((h_cond_1-h_marg_1)+(h_cond_2-h_marg_2))/2
                disc_c = discrepancy(lt1_c, lt2_c, 'mean')
                if it_pl is not None and lambda_lmmd_eff >0:
                    _, _, fs_ = model(xs)
                    _, _, fpl_c = model(xpl)
                    f_s_n = F.normalize(fs_, dim=1)
                    f_t_pl_n = F.normalize(fpl_c, dim=1)
                    if cached_gammas is None:
                        cached_gammas = suggest_mmd_gammas(f_s_n.detach(), f_t_pl_n.detach())
                    lmmd_loss = classwise_mmd_biased_weighted(
                        f_s_n, ys, f_t_pl_n, ypl, wpl,
                        num_classes=10, gammas=cached_gammas, min_count_per_class=2
                    )
                    loss_lmmd = lambda_lmmd_eff * lmmd_loss
                else:
                    loss_lmmd = ft_c.new_tensor(0.0)
                lossC = lambda_dis * disc_c  + loss_lmmd
                lossC.backward(); optim_G.step()
                sumC += lossC.item()
            countC += nC
            for p in model.c1.parameters(): p.requires_grad_(True)
            for p in model.c2.parameters(): p.requires_grad_(True)

            global_step += 1
        avg_lossA = sumA / max(1, countA)
        avg_lossB = sumB / max(1, countB)
        avg_lossC = sumC / max(1, countC)
        im_avg = im / max(1, countC)
        print(f"[Epoch {epoch:02d}] "
              f"lossA:{avg_lossA:.4f} lossB(last):{avg_lossB:.4f} lossC(last):{avg_lossC:.4f}" f"im_avg:{im_avg:.4f}")
        if epoch > 10 and im_avg < best_score:
            best_score = im_avg
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1:
            torch.save(model.state_dict(), os.path.join(save_dir, f"{tag}_epoch1.pth"))
            print(f"[SAVE] First epoch model saved: {tag}_epoch1.pth")


        # print("[INFO] Evaluating on target test set...")
        # test_ds = PKLDataset(tgt_test)
        # test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
        #
        # ACC_A, ACC_B, ACC_AVG = MCD_evaluate(model, test_loader, device)
        # print(f"ACC_A: {ACC_A:0.4f}, ACC_B: {ACC_B:0.4f}, ACC_AVG: {ACC_AVG:0.4f}")


    if best_state is not None:
        model.load_state_dict(best_state)
    final_path = os.path.join(save_dir, f"{tag}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[SAVE] Final/best model saved: {final_path}")

    return model

# ====== 入口 ======
if __name__ == "__main__":
    with open("/content/github/configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    bs = 64
    lr = 0.0002849852305416375
    wd = 8.939020441061186e-05
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = 15

    files = [185, 188, 191, 194]
    for file in files:

        src_path = '/content/datasets/DC_T197_RP.txt'
        tgt_path = '/content/datasets/HC_T{}_RP.txt'.format(file)
        tgt_test = '/content/datasets/HC_T{}_RP.txt'.format(file)

        print(f"[INFO] Loading HC_T{file} ...")

        for run_id in range(10):
            print(f"\n========== RUN {run_id} ==========")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 构建数据
            src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
            ps_loader = get_pseudo_dataloaders(tgt_path, bs)

            val_loader = DataLoader(PKLDataset(tgt_test), batch_size=bs, shuffle=False)

            # 构建模型
            model = Flexible_MCD(
                num_layers=num_layers, start_channels=sc, kernel_size=ksz, cnn_act='leakrelu',
                num_classes=10
            )

            # 训练（MCD）
            model = train_mcd(
                model, src_loader, tgt_loader, device,
                lmmd_start_epoch=3, ps_loader=ps_loader, max_lambda=0.5,
                num_epochs=num_epochs, lr_g=lr, lr_c=lr, weight_decay=wd,
                lambda_dis=1.0, nB=1, nC=2,
                save_dir="/content/drive/MyDrive/Masterarbeit/MCD_LMMD/T{file}",
                tag=f"RUN{run_id}"
            )

            print("[INFO] Evaluating on target test set...")
            test_ds = PKLDataset(tgt_test)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

            ACC_A, ACC_B, ACC_AVG= MCD_evaluate(model, test_loader, device)
            print(f"ACC_A: {ACC_A:.04f}, ACC_B: {ACC_B:.04f}, ACC_AVG: {ACC_AVG:.04f}")

