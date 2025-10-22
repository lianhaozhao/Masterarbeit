import os, json, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import optuna
import math
from models.Flexible_CNN_MCD import Flexible_MCD
from PKLDataset import PKLDataset
from utils.general_train_and_test import general_test_model
from models.get_no_label_dataloader import get_dataloaders,get_pseudo_dataloaders
import yaml
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_with_stats
from models.MMD import classwise_mmd_biased_weighted,suggest_mmd_gammas,infomax_loss_from_logits
from collections import deque


# MCD 的分歧度量
def discrepancy(logits1, logits2, reduction='mean'):
    p1, p2 = F.softmax(logits1, dim=1), F.softmax(logits2, dim=1)
    d = (p1 - p2).abs().sum(1)
    return d.mean() if reduction == 'mean' else d.sum() if reduction == 'sum' else d

def lambda_(epoch, num_epochs,max_lambda = 0.6):
    p = epoch / max(1, num_epochs - 1)
    return (2. / (1. + np.exp(-3 * p)) - 1.) * max_lambda
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
              num_epochs=15, lr_g=2e-4, lr_c=2e-4, weight_decay=0.0,
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
                pl_loader = DataLoader(pl_ds, batch_size=64, shuffle=True, drop_last=True)

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
            for _ in range(2):
                l1s, l2s, _ = model(xs)
                loss_src = F.cross_entropy(l1s, ys) + F.cross_entropy(l2s, ys)
                # 多样性正则（取最后一层线性权重；无 bias）
                W1 = model.c1.weight  # [C, hidden]
                W2 = model.c2.weight
                cos_sim = F.cosine_similarity(W1.flatten(), W2.flatten(), dim=0)
                loss_div = cos_sim.pow(2)

                lossA = loss_src + 1e-3 * loss_div
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
                loss_im_1, h_cond_1, h_marg_1 = infomax_loss_from_logits(lt1_c, T=1, marg_weight=1)
                loss_im_2, h_cond_2, h_marg_2 = infomax_loss_from_logits(lt2_c, T=1, marg_weight=1)
                im += ((h_cond_1-h_marg_1)+(h_cond_2-h_marg_2))/2
                loss_im = 0.5 *(loss_im_1 + loss_im_2)
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
                lossC = lambda_dis * disc_c + loss_im + loss_lmmd
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
        if epoch > 10 and im_avg < best_score:
            best_score = im_avg
            best_state = copy.deepcopy(model.state_dict())


        # print("[INFO] Evaluating on target test set...")
        # test_ds = PKLDataset(tgt_test)
        # test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
        #
        # ACC_A, ACC_B, ACC_AVG = MCD_evaluate(model, test_loader, device)
        # print(f"ACC_A: {ACC_A:0.4f}, ACC_B: {ACC_B:0.4f}, ACC_AVG: {ACC_AVG:0.4f}")


    if best_state is not None:
        model.load_state_dict(best_state)


    return model, avg_lossC



# 只搜对抗专属超参
def suggest_adv_only(trial):
    params = {
        "batch_size":    trial.suggest_categorical("batch_size", [64]),  # 可改成 [32, 48, 64]
        "learning_rate": trial.suggest_float("learning_rate",1e-4, 5e-4, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
    }
    return params


@torch.no_grad()
def infomax_unsup_score_from_loader(model, source_loader, target_loader, device,
                                    T=1.0, marg_weight=1.0, eps=1e-8):
    model.eval()
    probs = []
    for xb in target_loader:
        xb = xb.to(device)
        l1, l2, _ = model(xb)
        p = 0.5 * (F.softmax(l1 / T, 1) + F.softmax(l2 / T, 1))
        probs.append(p.detach().cpu())
    if not probs:
        return {"score": float("inf"), "domain_gap": 0.5}
    P = torch.cat(probs, dim=0).clamp_min(eps)
    mean_entropy = (-P * P.log()).sum(dim=1).mean().item()
    p_bar = P.mean(dim=0).clamp_min(eps)
    marginal_entropy = (-(p_bar * p_bar.log()).sum()).item()
    score = mean_entropy - marg_weight * marginal_entropy


    return {"score": float(score)}



def objective_adv_only(trial,
                       dataset_configs=None,
                       out_dir='../datasets/info_optuna_dann',
                       n_repeats=1,  # 每个数据集内部的重复次数
                       ):
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = 30
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = suggest_adv_only(trial)

    all_scores, all_gaps = [], []

    # 遍历多个数据集
    for dcfg in dataset_configs:
        for _ in np.arange(n_repeats):
            # 数据加载
            src_tr, tgt_tr = get_dataloaders(dcfg["source_train"], dcfg["target_train"],
                                             batch_size=p["batch_size"])
            ps_loader = get_pseudo_dataloaders(target_path=dcfg["target_train"],
                                             batch_size=p["batch_size"])

            # 模型
            model = Flexible_MCD(
                num_layers=num_layers, start_channels=sc, kernel_size=ksz, cnn_act='leakrelu',
                num_classes=10
            )

            # 训练（MCD）
            model, avg_lossC = train_mcd(
                model, src_tr, tgt_tr, device,
                lmmd_start_epoch=3, ps_loader=ps_loader, max_lambda=0.5,
                num_epochs=num_epochs,
                lr_g=p["learning_rate"],
                lr_c=p["learning_rate"],
                weight_decay=p["weight_decay"],
                lambda_dis=1.0, nB=2, nC=1
            )

            metrics = infomax_unsup_score_from_loader(model, src_tr, tgt_tr, device,
                                                      T=1.0, marg_weight=1.0)

            all_scores.append(metrics["score"])
            all_gaps.append([avg_lossC])

    # ======== 汇总统计 ========
    mean_score, std_score = np.mean(all_scores), np.std(all_scores, ddof=1)
    mean_gap, std_gap     = np.mean(all_gaps), np.std(all_gaps, ddof=1)

    alpha = 1.0
    robust = mean_score + mean_gap + alpha * std_gap

    trial.set_user_attr("mean_score", float(mean_score))
    trial.set_user_attr("std_score",  float(std_score))
    trial.set_user_attr("mean_gap",   float(mean_gap))
    trial.set_user_attr("std_gap",    float(std_gap))
    trial.set_user_attr("robust",     float(robust))

    print(f"[Trial {trial.number}] mean_score={mean_score:.4f}±{std_score:.4f} | "
          f"mean_gap={mean_gap:.3f}±{std_gap:.3f} | robust={robust:.4f}")

    return float(mean_score), float(mean_gap)



# ========= 主程序 =========
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 直接搜索对抗专属超参
    sampler = optuna.samplers.NSGAIISampler()
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    study.optimize(lambda t: objective_adv_only(
        t,
        [  # 多个数据集组合
            {
                "source_train": "../datasets/source/train/DC_T197_RP.txt",
                "target_train": "../datasets/target/train/HC_T185_RP.txt",
            },
            {
                "source_train": "../datasets/source/train/DC_T197_RP.txt",
                "target_train": "../datasets/target/train/HC_T185_RP.txt",
            },
        ],
        out_dir  = '../datasets/info_optuna_dann',
    ), n_trials=50)

    pareto = study.best_trials
    print("Pareto size:", len(pareto))
    for t in pareto[:5]:
        print("trial#", t.number, "values=", t.values, "params=", t.params)


    def scalarize(t,  beta=0.5):
        score, gap = t.values
        return score + beta * gap

    chosen = min(pareto, key=lambda t: scalarize(t,  1))
    best = {
        "chosen_trial_number": chosen.number,
        "chosen_params": chosen.params,
        "chosen_values": chosen.values
    }
    out_dir = '../datasets/info_optuna_dann'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "best_adv_only.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("[DANN-AdvOnly] Best:", best)
