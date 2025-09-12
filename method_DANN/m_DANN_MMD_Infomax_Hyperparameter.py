import os, json, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import optuna
import math
from models.Flexible_DANN_LMMD import Flexible_DANN
from PKLDataset import PKLDataset
from utils.general_train_and_test import general_test_model
from models.get_no_label_dataloader import get_dataloaders
import yaml
from models.MMD import mmd2_unconditional,suggest_mmd_gammas
from collections import deque


def adam_param_groups(model, weight_decay):
    """对 BN/偏置不做 weight decay"""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
# InfoMax (target domain)
@torch.no_grad()
def _safe_mean_prob(p, eps=1e-8):
    p = p.clamp_min(eps)
    return p / p.sum(dim=1, keepdim=True)

def entropy_mean(p, eps=1e-8):
    # E_x[ H(p_x) ]
    p = p.clamp_min(eps)
    return (-p * p.log()).sum(dim=1).mean()

def entropy_marginal(p, eps=1e-8):
    # H( E_x[p_x] )
    p_bar = p.mean(dim=0)
    p_bar = p_bar.clamp_min(eps)
    return -(p_bar * p_bar.log()).sum()

def infomax_loss_from_logits(logits, T=1.0, marg_weight=1.0):
    """
        依据 “信息最大化” (InfoMax) 思想，从分类 logits 计算无监督正则项：
        I(z; ŷ) = H(ŷ) - H(ŷ|z)。
        训练中最小化的目标为：
            L = H(ŷ|z) - w * H(ŷ)
        其中 H(ŷ|z) 是条件熵（鼓励单样本预测更自信），H(ŷ) 是边际熵（鼓励整体类别使用均衡，防止塌缩）。

        参数
        ----
        logits : Tensor
            分类头输出的未归一化得分，形状 [B, C]。
        T : float, 默认 1.0
            Softmax 温度。T > 1 使分布变“软”（置信度下降），T < 1 使分布变“尖”（置信度上升）。
            会同时影响条件熵与边际熵的数值。
        marg_weight : float, 默认 1.0
            边际熵权重 w。数值越大，越强烈地抑制“塌缩到单一类别”的解。
            常见范围 0.5 ~ 2.0，可据验证集曲线调参。

        返回
        ----
        loss : Tensor (标量，requires_grad=True)
            最小化目标：H(ŷ|z) - w * H(ŷ)。用于反向传播。
        h_cond_detached : Tensor (标量，no grad)
            条件熵 H(ŷ|z) 的经验估计（E_x[-∑_c p(c|x) log p(c|x)]）。仅用于日志监控。
        h_marg_detached : Tensor (标量，no grad)
            边际熵 H(ŷ) 的经验估计，其中 ŷ 的边际分布为 p̄ = E_x[p(·|x)]。
            数学形式：H(ŷ) = -∑_c p̄_c log p̄_c。仅用于日志监控。

        """
    # I(z;ŷ) = H(ŷ) - H(ŷ|z); minimiere  H(ŷ|z) - w * H(ŷ)
    p = F.softmax(logits / T, dim=1)  # [B, C], Klassenwahrscheinlichkeiten pro Beispiel
    h_cond = entropy_mean(p)  # Skalar, Schätzung der bedingten Entropie
    h_marg = entropy_marginal(p)  # Schätzung der marginalen Entropie
    return h_cond - marg_weight * h_marg, h_cond.detach(), h_marg.detach()

# DANN Lambda Scheduling (GRL only)
def dann_lambda(epoch, num_epochs, max_lambda=0.5):
    p = epoch / max(1, num_epochs-1)
    return (2.0 / (1.0 + np.exp(-5 * p)) - 1.0) * max_lambda

# Baseline weight of MMD (multiplied by quality gate to get final weight)
def mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=5):
    if epoch < start_epoch:
        return 0.0
    p = (epoch - start_epoch) / max(1, (num_epochs - 1 - start_epoch))
    return (2.0 / (1.0 + np.exp(-3 * p)) - 1.0) * max_lambda

# Training
def train_dann_infomax_lmmd(model,
                            source_loader, target_loader,src_va,
                            optimizer, criterion_cls, criterion_domain,
                            device, num_epochs=20,
                            scheduler=None,
                            # InfoMax
                            im_T=1.0, im_weight=0.5, im_marg_w=1.0,
                            # Gating
                            grl_lambda_max=1,max_lambda = 0.6,
                            mmd_start_epoch=5,
                            ):
    # Hyperparameters related to early stopping
    W = 4
    GAP_TH = 0.05  # DomAcc distance threshold of 0.5 (the smaller the better alignment)
    PATIENCE = 3  # Stop after several rounds without improvement

    # Track Cache & Best Recording
    gap_hist = deque(maxlen=W)
    best_score = float('inf')
    best_state = None
    plateau_best_score = float('inf')
    best_val = float("inf")
    wait = 0
    def eval_on_source():
        model.eval()
        total, loss_sum = 0, 0.0
        with torch.no_grad():
            for xb, yb in src_va:
                xb, yb = xb.to(device), yb.to(device)
                cls_out, _, _ = model(xb, grl=False)
                v = criterion_cls(cls_out, yb)
                loss_sum += v.item() * xb.size(0)
                total += xb.size(0)
        return loss_sum / max(1, total)

    for epoch in range(num_epochs):
        cached_gammas = None
        # 1) epoch training
        model.train()
        it_src = iter(source_loader)
        it_tgt = iter(target_loader)
        len_src, len_tgt = len(source_loader), len(target_loader)
        num_iters = max(len_src, len_tgt)

        cls_loss_sum = dom_loss_sum = mmd_loss_sum = im_loss_sum = 0.0
        tot_loss_sum = 0.0
        tot_target_samples = tot_cls_samples = tot_dom_samples = 0
        dom_correct_src = dom_correct_tgt = 0
        dom_total_src = dom_total_tgt = 0

        for _ in range(num_iters):
            try:
                src_x, src_y = next(it_src)
            except StopIteration:
                it_src = iter(source_loader);
                src_x, src_y = next(it_src)
            try:
                tgt_x = next(it_tgt)
            except StopIteration:
                it_tgt = iter(target_loader);
                tgt_x = next(it_tgt)
            if isinstance(tgt_x, (tuple, list)): tgt_x = tgt_x[0]

            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            # Put λ only into GRL
            model.lambda_ = float(dann_lambda(epoch, num_epochs, max_lambda=grl_lambda_max))
            cls_out_src, dom_out_src, feat_src = model(src_x, grl=True)
            cls_out_tgt, dom_out_tgt, feat_tgt = model(tgt_x, grl=True)

            # 1) Source classification
            loss_cls = criterion_cls(cls_out_src, src_y)

            # 2) Domain confrontation
            dom_label_src = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            dom_label_tgt = torch.ones(tgt_x.size(0), dtype=torch.long, device=device)
            loss_dom = (
                               criterion_domain(dom_out_src, dom_label_src) * src_x.size(0)
                               + criterion_domain(dom_out_tgt, dom_label_tgt) * tgt_x.size(0)
                       ) / (src_x.size(0) + tgt_x.size(0))

            # 3) InfoMax
            loss_im, h_cond, h_marg = infomax_loss_from_logits(cls_out_tgt, T=im_T, marg_weight=im_marg_w)
            loss_im = im_weight * loss_im

            # 4) MMD
            lambda_mmd_eff = mmd_lambda(epoch, num_epochs, max_lambda=max_lambda, start_epoch=mmd_start_epoch)
            _, _, feat_src_plain = model(src_x, grl=False)
            _, _, feat_tgt_plain = model(tgt_x, grl=False)
            fs = F.normalize(feat_src_plain, dim=1)
            ft = F.normalize(feat_tgt_plain, dim=1)
            if cached_gammas is None:
                cached_gammas = suggest_mmd_gammas(fs.detach(), ft.detach())
            gammas = cached_gammas
            loss_mmd = mmd2_unconditional(fs, ft, gammas)

            loss = loss_cls + loss_dom + loss_im + loss_mmd * lambda_mmd_eff

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # ------ Statistics ------
            cls_loss_sum += loss_cls.item() * src_x.size(0)
            dom_loss_sum += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            im_loss_sum += loss_im.item() * (tgt_x.size(0))
            mmd_loss_sum += loss_mmd.item() * src_x.size(0)
            tot_loss_sum += loss.item() * (src_x.size(0) + tgt_x.size(0))
            tot_cls_samples += src_x.size(0)
            tot_dom_samples += (src_x.size(0) + tgt_x.size(0))
            tot_target_samples += tgt_x.size(0)

            dom_correct_src += (dom_out_src.argmax(1) == dom_label_src).sum().item()
            dom_total_src += dom_label_src.size(0)
            dom_correct_tgt += (dom_out_tgt.argmax(1) == dom_label_tgt).sum().item()
            dom_total_tgt += dom_label_tgt.size(0)

        # ---- Epoch Log ----
        avg_cls = cls_loss_sum / max(1, tot_cls_samples)
        avg_dom = dom_loss_sum / max(1, tot_dom_samples)
        avg_im = im_loss_sum / max(1, tot_target_samples)
        avg_mmd = mmd_loss_sum / max(1, tot_cls_samples)
        avg_tot = tot_loss_sum / max(1, tot_dom_samples)
        acc_src = dom_correct_src / max(1, dom_total_src)
        acc_tgt = dom_correct_tgt / max(1, dom_total_tgt)
        dom_acc = 0.5 * (acc_src + acc_tgt)
        gap = abs(dom_acc - 0.5)
        if scheduler is not None: scheduler.step()

        val_loss = eval_on_source()
        if epoch > 12:
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= 3:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val



# 只搜对抗专属超参
def suggest_adv_only(trial):
    params = {
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64]),  # 可改成 [32, 48, 64]
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-7, 5e-3, log=True),
        "grl_lambda_max":    trial.suggest_categorical("grl_lambda_max", [0.5,0.6,0.7,0.8,0.9]),
        "mmd_lambda_max":   trial.suggest_float("mmd_lambda_max", 0.35, 0.8, log=True),
        "im_weight":     trial.suggest_categorical("im_weight", [0.5,0.6,0.7,0.8,0.9]),
        "im_T":          trial.suggest_categorical("im_T", [1.0])
    }
    return params


@torch.no_grad()
def infomax_unsup_score_from_loader(model, source_loader, target_loader, device,
                                    T=1.0, marg_weight=1.0, eps=1e-8):
    model.eval()
    probs = []
    for xb in target_loader:
        xb = xb.to(device)
        cls_out, _, _ = model(xb, grl=False)
        p = F.softmax(cls_out / T, dim=1)
        probs.append(p.detach().cpu())
    if not probs:
        return {"score": float("inf"), "domain_gap": 0.5}
    P = torch.cat(probs, dim=0).clamp_min(eps)
    mean_entropy = (-P * P.log()).sum(dim=1).mean().item()
    p_bar = P.mean(dim=0).clamp_min(eps)
    marginal_entropy = (-(p_bar * p_bar.log()).sum()).item()
    score = mean_entropy - marg_weight * marginal_entropy

    # 域不可分性（gap 越小越好）
    tot, correct = 0, 0
    for (xs, _), xt in zip(source_loader, target_loader):
        xs, xt = xs.to(device), xt.to(device)
        _, dom_s, _= model(xs)
        _, dom_t, _= model(xt)
        pred_s = dom_s.argmax(dim=1); pred_t = dom_t.argmax(dim=1)
        correct += (pred_s == 0).sum().item()
        correct += (pred_t == 1).sum().item()
        tot += xs.size(0) + xt.size(0)
    domain_acc = (correct / max(1, tot)) if tot > 0 else 1.0
    domain_gap = abs(domain_acc - 0.5)
    return {"score": float(score), "domain_gap": float(domain_gap)}



def objective_adv_only(trial,
                       source_train='../datasets/source/train/DC_T197_RP.txt',
                       source_val  ='../datasets/source/test/DC_T197_RP.txt',
                       target_train='../datasets/target/train/HC_T185_RP.txt',
                       out_dir     ='../datasets/info_optuna_dann',
                       ):
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = 30
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = suggest_adv_only(trial)

    # 数据（batch 固定）
    src_tr, tgt_tr = get_dataloaders(source_train, target_train, batch_size=p["batch_size"])
    src_va = DataLoader(PKLDataset(source_val), batch_size=p["batch_size"], shuffle=False)

    # 模型（结构固定）
    model = Flexible_DANN(num_layers=num_layers,
                          start_channels=sc,
                          kernel_size=ksz,
                          num_classes=10,
                          lambda_=1).to(device)

    optimizer = torch.optim.Adam( adam_param_groups(model, p["weight_decay"]), lr=p["learning_rate"], betas=(0.9, 0.999), eps=1e-8 )
    # 调度器（Cosine；与训练轮数一致）
    max_epochs = num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=p["learning_rate"] * 0.1
    )
    c_cls = nn.CrossEntropyLoss()
    c_dom = nn.CrossEntropyLoss()


    best_val = train_dann_infomax_lmmd(model, src_tr, tgt_tr,src_va,
                optimizer, c_cls, c_dom, device,
                num_epochs=num_epochs,
                scheduler=scheduler,
                # InfoMax Hyperparameters
                im_T=p["im_T"], im_weight=p["im_weight"], im_marg_w=1.0,
                grl_lambda_max=p["grl_lambda_max"],max_lambda=p["mmd_lambda_max"],
                mmd_start_epoch=4,
                                       )

    metrics = infomax_unsup_score_from_loader(model, src_tr, tgt_tr, device,
                                              T=1.0, marg_weight=1.0)
    trial.set_user_attr('best_val', best_val)
    trial.set_user_attr("score", metrics["score"])
    trial.set_user_attr("domain_gap", metrics["domain_gap"])

    print(f"[InfoMax] score={metrics['score']:.4f} | "
          f"gap={metrics['domain_gap']:.3f}")

    # 记录 trial
    rec = {
        "trial": trial.number, **p,
        "best_val_loss": float(best_val),
        "score": metrics["score"],
        "domain_gap": metrics['domain_gap']
    }
    path = os.path.join(out_dir, "trials_adv_only.json")
    all_rec = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f: all_rec = json.load(f)
        except Exception: all_rec = []
    all_rec.append(rec)
    with open(path, "w") as f: json.dump(all_rec, f, indent=2)

    # 2目标：最小化 InfoMax 分数 最小化 domain_gap
    return float(metrics["score"]), float(metrics["domain_gap"])


# ========= 主程序 =========
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 直接搜索对抗专属超参
    sampler = optuna.samplers.NSGAIISampler()
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    study.optimize(lambda t: objective_adv_only(
        t,
        source_train='../datasets/source/train/DC_T197_RP.txt',
        source_val  ='../datasets/source/test/DC_T197_RP.txt',
        target_train='../datasets/target/train/HC_T185_RP.txt',
        out_dir     ='../datasets/info_optuna_dann',
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
