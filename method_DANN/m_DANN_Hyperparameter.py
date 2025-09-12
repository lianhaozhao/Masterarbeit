import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna
import math
from models.Flexible_DANN import Flexible_DANN
from PKLDataset import PKLDataset
from utils.general_train_and_test import general_test_model
from models.get_no_label_dataloader import get_dataloaders
import yaml


def set_seed(seed=44):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def dann_lambda(epoch, num_epochs, max_lambda=1.0):
    p = epoch / max(1, num_epochs - 1)
    return (2.0 / (1.0 + np.exp(-7 * p)) - 1.0) * max_lambda

def train_dann(model, source_loader, target_loader, val_loader,
               optimizer, scheduler, device,
               max_epochs=30, lambda_max=1.0, early_patience=5):
    ce = nn.CrossEntropyLoss()
    best_val, best_state, wait = float('inf'), None, 0

    for epoch in range(max_epochs):
        model.train()
        for (src_x, src_y), tgt_x in zip(source_loader, target_loader):
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            model.lambda_ = float(dann_lambda(epoch, max_epochs, max_lambda=lambda_max))

            cls_out_src, dom_out_src = model(src_x)
            _,          dom_out_tgt = model(tgt_x)

            loss_cls = ce(cls_out_src, src_y)

            # 域损失（样本数加权）

            bs_src, bs_tgt = src_x.size(0), tgt_x.size(0)
            dom_label_src = torch.zeros(bs_src, dtype=torch.long, device=device)
            dom_label_tgt = torch.ones(bs_tgt, dtype=torch.long, device=device)
            loss_dom_src  = ce(dom_out_src, dom_label_src)
            loss_dom_tgt  = ce(dom_out_tgt, dom_label_tgt)
            loss_dom = (loss_dom_src * bs_src + loss_dom_tgt * bs_tgt) / (bs_src + bs_tgt)

            loss = loss_cls + loss_dom

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # 源域验证（UDA 合规）
        model.eval()
        with torch.no_grad():
            v_total, v_sum = 0, 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                cls_out_src, _ = model(xb,grl=False)
                v = ce(cls_out_src, yb)
                v_total += xb.size(0); v_sum += v.item() * xb.size(0)
            val_loss = v_sum / max(1, v_total)

        if scheduler is not None: scheduler.step()

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= early_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val

# 只搜对抗专属超参
def suggest_adv_only(trial):
    params = {
        "batch_size":    trial.suggest_categorical("batch_size", [32]),  # 你也可改成 [32, 48, 64]
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-7, 5e-3, log=True),
        "lambda_max":    trial.suggest_categorical("lambda_max", [1]),
    }
    return params


@torch.no_grad()
def infomax_unsup_score_from_loader(model, source_loader, target_loader, device,
                                    T=1.0, marg_weight=1.0, eps=1e-8):
    """
    以 InfoMax + 域不可分性合成无监督分数
    Score = mean_entropy - w * marginal_entropy
    同时返回 domain_gap = |domain_acc - 0.5|
    """
    model.eval()
    probs = []
    #  1) 条件熵 & 边际熵
    for xb in target_loader:
        xb = xb.to(device)
        cls_out, _ = model(xb, grl=False)                 # DANN forward: (cls_out, dom_out)
        p = F.softmax(cls_out / T, dim=1)      # [B,C]
        probs.append(p.detach().cpu())
    if not probs:
        return {"score": float("inf"), "domain_gap": 0.5}

    P = torch.cat(probs, dim=0)               # [N,C]
    P = P.clamp_min(eps)

    # 条件熵: E_x[ -∑ p log p ]
    mean_entropy = (-P * P.log()).sum(dim=1).mean().item()
    # 边际熵: H( E_x[p] )
    p_bar = P.mean(dim=0).clamp_min(eps)
    marginal_entropy = (-(p_bar * p_bar.log()).sum()).item()

    score = mean_entropy - marg_weight * marginal_entropy

    # 2) 域判别准确率，转 gap
    tot, correct = 0, 0
    for (xs, _), xt in zip(source_loader, target_loader):
        xs, xt = xs.to(device), xt.to(device)
        _, dom_s = model(xs)
        _, dom_t = model(xt)
        pred_s = dom_s.argmax(dim=1)
        pred_t = dom_t.argmax(dim=1)
        correct += (pred_s == 0).sum().item()
        correct += (pred_t == 1).sum().item()
        tot += xs.size(0) + xt.size(0)
    domain_acc = (correct / max(1, tot)) if tot > 0 else 1.0
    domain_gap = abs(domain_acc - 0.5)  # 越接近 0 越好

    return {"score": float(score),
            "domain_gap": float(domain_gap)}



def objective_adv_only(trial,
                       source_train='../datasets/source/train/DC_T197_RP.txt',
                       source_val  ='../datasets/source/test/DC_T197_RP.txt',
                       target_train='../datasets/target/train/HC_T185_RP.txt',
                       out_dir     ='../datasets/info_optuna_dann',
                       ):
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['Baseline']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = cfg['num_epochs']
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


    best_val = train_dann(model, src_tr, tgt_tr, src_va, optimizer, scheduler, device,
                          max_epochs=max_epochs, lambda_max=p["lambda_max"], early_patience=4)

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
    # set_seed(44)
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
    ), n_trials=40)

    pareto = study.best_trials
    print("Pareto size:", len(pareto))
    for t in pareto[:5]:
        print("trial#", t.number, "values=", t.values, "params=", t.params)


    def scalarize(t,  beta=0.5):
        score, gap = t.values
        return score + beta * gap

    chosen = min(pareto, key=lambda t: scalarize(t,  0.5))
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
