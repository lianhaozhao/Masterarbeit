import os, json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna
import yaml
from models.Flexible_DANN_LMMD import Flexible_DANN
from PKLDataset import PKLDataset
from models.get_no_label_dataloader import get_dataloaders


def adam_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]

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

# ====== 多核 RBF MMD ======
def _pairwise_sq_dists(a, b):
    a2 = (a*a).sum(dim=1, keepdim=True)
    b2 = (b*b).sum(dim=1, keepdim=True).t()
    return a2 + b2 - 2 * (a @ b.t())

def _mk_kernel(a, b, gammas):
    d2 = _pairwise_sq_dists(a, b).clamp_min(0)
    k = 0.0
    M = max(1, len(gammas))
    for g in gammas:
        k = k + torch.exp(-float(g) * d2)
    return k / M

def mmd2_unconditional(a, b, gammas):
    Kaa = _mk_kernel(a, a, gammas); Kab = _mk_kernel(a, b, gammas); Kbb = _mk_kernel(b, b, gammas)
    # 有偏估计，稳定即可
    e_aa = Kaa.mean(); e_bb = Kbb.mean(); e_ab = Kab.mean()
    return (e_aa + e_bb - 2 * e_ab).clamp_min(0.0)

@torch.no_grad()
def suggest_mmd_gammas(fs, ft, scales=(0.25,0.5,1,2,4)):
    x = torch.cat([fs.detach(), ft.detach()], dim=0)
    xi, xj = x.unsqueeze(1), x.unsqueeze(0)
    d2 = (xi - xj).pow(2).sum(-1).flatten()
    m = d2.clamp_min(1e-12).median()
    g0 = (1.0 / (2.0 * m)).item()
    return [s * g0 for s in scales]

# 无监督评估：InfoMax + 域不可分性
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

def train_dann_mmd(model, source_loader, target_loader,
                   optimizer, scheduler, device,
                   max_epochs=30,
                   grl_lambda_max=1.0,
                   mmd_lambda_max=0.35,
                   mmd_start_epoch=5,
                   early_patience=5,
                   source_val_loader=None):
    best_val, best_state, wait = float('inf'), None, 0
    c_cls = nn.CrossEntropyLoss()
    c_dom = nn.CrossEntropyLoss()

    def eval_on_source():
        model.eval()
        total, s = 0, 0.0
        with torch.no_grad():
            for xb, yb in source_val_loader:
                xb, yb = xb.to(device), yb.to(device)
                cls_out, _, _ = model(xb, grl=False)
                v = c_cls(cls_out, yb)
                total += xb.size(0); s += v.item() * xb.size(0)
        return s / max(1, total)

    for epoch in range(max_epochs):
        model.train()
        it_t = iter(target_loader)
        cached_gammas = None

        for (src_x, src_y) in source_loader:
            try:
                tgt_x = next(it_t)
            except StopIteration:
                it_t = iter(target_loader); tgt_x = next(it_t)

            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            # 1) 对抗
            model.lambda_ = float(dann_lambda(epoch, max_epochs, max_lambda=grl_lambda_max))
            cls_src, dom_src, _ = model(src_x, grl=True)
            _,       dom_tgt, _ = model(tgt_x, grl=True)

            loss_cls = c_cls(cls_src, src_y)
            bs_s, bs_t = src_x.size(0), tgt_x.size(0)
            dom_label_s = torch.zeros(bs_s, dtype=torch.long, device=device)
            dom_label_t = torch.ones(bs_t, dtype=torch.long, device=device)
            loss_dom = (c_dom(dom_src, dom_label_s) * bs_s + c_dom(dom_tgt, dom_label_t) * bs_t) / (bs_s + bs_t)

            # 2)
            lambda_mmd = mmd_lambda(epoch, max_epochs, max_lambda=mmd_lambda_max, start_epoch=mmd_start_epoch)
            _, _, fs_plain = model(src_x, grl=False)
            _, _, ft_plain = model(tgt_x, grl=False)
            fs = F.normalize(fs_plain, dim=1)
            ft = F.normalize(ft_plain, dim=1)

            if cached_gammas is None:
                cached_gammas = suggest_mmd_gammas(fs.detach(), ft.detach())
            loss_mmd = mmd2_unconditional(fs, ft, cached_gammas)

            loss = loss_cls + loss_dom + lambda_mmd * loss_mmd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # 源域“验证”
        val_loss = eval_on_source()
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

# ====== 超参搜索空间（含 MMD 退火超参） ======
def suggest_mmd_only(trial):
    return {
        "batch_size":       trial.suggest_categorical("batch_size", [32]),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay":     trial.suggest_float("weight_decay", 1e-7, 5e-3, log=True),
        "grl_lambda_max":   trial.suggest_categorical("grl_lambda_max", [0.8,1.0]),
        "mmd_lambda_max":   trial.suggest_float("mmd_lambda_max", 0.15, 0.5, log=True),
    }

# ====== 目标函数 ======
def objective_mmd_only(trial,
                       source_train='../datasets/source/train/DC_T197_RP.txt',
                       source_val  ='../datasets/source/test/DC_T197_RP.txt',
                       target_train='../datasets/target/train/HC_T185_RP.txt',
                       out_dir     ='../datasets/info_optuna_mmd'):
    with open("../configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)["Baseline"]
    num_layers = cfg["num_layers"]; ksz = cfg["kernel_size"]; sc = cfg["start_channels"]
    num_epochs = cfg["num_epochs"]

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = suggest_mmd_only(trial)

    # data
    src_tr, tgt_tr = get_dataloaders(source_train, target_train, batch_size=p["batch_size"])
    src_va = DataLoader(PKLDataset(source_val), batch_size=p["batch_size"], shuffle=False)

    # model
    model = Flexible_DANN(num_layers=num_layers,
                          start_channels=sc,
                          kernel_size=ksz,
                          num_classes=10,
                          lambda_=1).to(device)

    # opt & sched
    optimizer = torch.optim.Adam(
        adam_param_groups(model, p["weight_decay"]),
        lr=p["learning_rate"], betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=p["learning_rate"] * 0.1
    )

    # train
    best_val = train_dann_mmd(
        model, src_tr, tgt_tr, optimizer, scheduler, device,
        max_epochs=num_epochs,
        grl_lambda_max=p["grl_lambda_max"],
        mmd_lambda_max=p["mmd_lambda_max"],
        early_patience=3,
        source_val_loader=src_va
    )

    # unsup metrics（与 DANN 版一致）
    metrics = infomax_unsup_score_from_loader(model, src_tr, tgt_tr, device,
                                              T=1.0, marg_weight=1.0)
    trial.set_user_attr("best_val", best_val)
    trial.set_user_attr("score", metrics["score"])
    trial.set_user_attr("domain_gap", metrics["domain_gap"])
    print(f"[InfoMax] score={metrics['score']:.4f} | gap={metrics['domain_gap']:.3f}")

    # 记录 trial
    rec = {"trial": trial.number, **p,
           "best_val_loss": float(best_val),
           "score": metrics["score"],
           "domain_gap": metrics["domain_gap"]}
    path = os.path.join(out_dir, "trials_mmd_only.json")
    all_rec = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f: all_rec = json.load(f)
        except Exception: all_rec = []
    all_rec.append(rec)
    with open(path, "w") as f: json.dump(all_rec, f, indent=2)

    # 多目标：最小化 InfoMax 分数 & 最小化域 gap
    return float(metrics["score"]), float(metrics["domain_gap"])


if __name__ == "__main__":

    sampler = optuna.samplers.NSGAIISampler()
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    study.optimize(lambda t: objective_mmd_only(
        t,
        source_train='../datasets/source/train/DC_T197_RP.txt',
        source_val  ='../datasets/source/test/DC_T197_RP.txt',
        target_train='../datasets/target/train/HC_T185_RP.txt',
        out_dir     ='../datasets/info_optuna_mmd',
    ), n_trials=40)

    pareto = study.best_trials
    print("Pareto size:", len(pareto))
    for tr in pareto[:5]:
        print("trial#", tr.number, "values=", tr.values, "params=", tr.params)

    def scalarize(tr, beta=0.5):
        score, gap = tr.values
        return score + beta * gap

    chosen = min(pareto, key=lambda tr: scalarize(tr, 0.5))
    best = {
        "chosen_trial_number": chosen.number,
        "chosen_params": chosen.params,
        "chosen_values": chosen.values
    }
    os.makedirs('../datasets/info_optuna_mmd', exist_ok=True)
    with open(os.path.join('../datasets/info_optuna_mmd', "best_mmd_only.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("[DANN+MMD] Best:", best)
