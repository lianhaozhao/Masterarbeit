import copy, math, random, os, json
import yaml
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from models.Flexible_DANN_LMMD import Flexible_DANN
from PKLDataset import PKLDataset
from models.get_no_label_dataloader import get_dataloaders, get_pseudo_dataloaders
from utils.general_train_and_test import general_test_model
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_with_stats
from models.MMD import classwise_mmd_biased_weighted, suggest_mmd_gammas



def dann_lambda(epoch, num_epochs, max_lambda=0.5):
    p = epoch / max(1, num_epochs-1)
    return (2.0 / (1.0 + np.exp(-5 * p)) - 1.0) * max_lambda

def mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=5):
    if epoch < start_epoch:
        return 0.0
    p = (epoch - start_epoch) / max(1, (num_epochs - 1 - start_epoch))
    return (2.0 / (1.0 + np.exp(-3 * p)) - 1.0) * max_lambda

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


# InfoMax

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



def train_dann_lmmd(model,
                    source_loader, source_val_loader ,target_loader,ps_loader,
                    optimizer, criterion_cls, criterion_domain,
                    device, num_epochs=20, num_classes=10,batchsize=64,
                    pseudo_thresh=0.95,
                    scheduler=None,
                    grl_lambda_max = 1,
                    lmmd_start_epoch=5, lmmd_t=1,
                    lmmd_lambda_max=0.1):
    # Early stopping 参数
    def eval_on_source():
        model.eval()
        total, s = 0, 0.0
        with torch.no_grad():
            for xb, yb in source_val_loader:
                xb, yb = xb.to(device), yb.to(device)
                cls_out, _, _ = model(xb, grl=False)
                v = criterion_cls(cls_out, yb)
                total += xb.size(0); s += v.item() * xb.size(0)
        return s / max(1, total)


    best_state = None
    best_val = float("inf")
    wait = 0

    for epoch in range(num_epochs):
        # 1) Pseudo-labeling
        pl_loader = None
        cached_gammas = None
        pseudo_x = torch.empty(0)
        pseudo_y = torch.empty(0, dtype=torch.long)
        pseudo_w = torch.empty(0)
        cov = margin_mean = 0.0
        if epoch >= lmmd_start_epoch:
            pseudo_x, pseudo_y, pseudo_w, stats = generate_pseudo_with_stats(
                model, ps_loader, device, threshold=pseudo_thresh, T=lmmd_t
            )
            kept, total = stats["kept"], stats["total"]
            cov, margin_mean = stats["coverage"], stats["margin_mean"]
            if kept > 0:
                pl_ds = TensorDataset(pseudo_x, pseudo_y, pseudo_w)
                pl_loader = DataLoader(pl_ds, batch_size=batchsize, shuffle=True,drop_last=True)

        # 2) Gated LMMD weights
        lambda_mmd_eff = mmd_lambda(epoch, num_epochs,
                                    max_lambda=lmmd_lambda_max,
                                    start_epoch=lmmd_start_epoch)

        # 3) Training loop
        model.train()
        it_src = iter(source_loader)
        it_tgt = iter(target_loader)
        it_pl  = iter(pl_loader) if pl_loader is not None else None
        len_src, len_tgt = len(source_loader), len(target_loader)
        len_pl = len(pl_loader) if pl_loader is not None else 0
        num_iters = max(len_src, len_tgt, len_pl) if len_pl > 0 else max(len_src, len_tgt)

        for _ in range(num_iters):
            try: src_x, src_y = next(it_src)
            except StopIteration:
                it_src = iter(source_loader); src_x, src_y = next(it_src)
            try: tgt_x = next(it_tgt)
            except StopIteration:
                it_tgt = iter(target_loader); tgt_x = next(it_tgt)
            if it_pl is not None:
                try: tgt_pl_x, tgt_pl_y, tgt_pl_w = next(it_pl)
                except StopIteration:
                    it_pl = iter(pl_loader); tgt_pl_x, tgt_pl_y, tgt_pl_w = next(it_pl)
            else:
                tgt_pl_x = tgt_pl_y = tgt_pl_w = None

            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)
            if tgt_pl_x is not None:
                tgt_pl_x = tgt_pl_x.to(device)
                tgt_pl_y = tgt_pl_y.to(device)
                tgt_pl_w = tgt_pl_w.to(device)

            # Forward
            model.lambda_ = float(dann_lambda(epoch, num_epochs, max_lambda=grl_lambda_max))
            cls_out_src, dom_out_src, feat_src = model(src_x, grl=True)
            _, dom_out_tgt, feat_tgt = model(tgt_x, grl=True)

            # 1) Classification loss
            loss_cls = criterion_cls(cls_out_src, src_y)

            # 2) Domain loss
            dom_label_src = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            dom_label_tgt = torch.ones(tgt_x.size(0),  dtype=torch.long, device=device)
            loss_dom = (
                criterion_domain(dom_out_src, dom_label_src) * src_x.size(0)
                + criterion_domain(dom_out_tgt, dom_label_tgt) * tgt_x.size(0)
            ) / (src_x.size(0) + tgt_x.size(0))

            # 3) LMMD
            if tgt_pl_x is not None and lambda_mmd_eff > 0:
                _, _, feat_source = model(src_x, grl=False)
                _, _, feat_tgt_pl = model(tgt_pl_x, grl=False)
                if cached_gammas is None:
                    cached_gammas = suggest_mmd_gammas(feat_source.detach(), feat_tgt_pl.detach())
                loss_lmmd = classwise_mmd_biased_weighted(
                    feat_source, src_y, feat_tgt_pl, tgt_pl_y, tgt_pl_w,
                    num_classes=num_classes, gammas=cached_gammas,
                    min_count_per_class=3
                )
                loss_lmmd = lambda_mmd_eff * loss_lmmd
            else:
                loss_lmmd = src_x.new_tensor(0.0)

            loss = loss_cls + loss_dom + loss_lmmd

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
            if wait >= 3:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val



# ====================== Optuna 超参搜索 ======================

def suggest_lmmd_only(trial):
    return {
        "batch_size":        trial.suggest_categorical("batch_size", [32, 64]),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay":      trial.suggest_float("weight_decay", 1e-7, 5e-3, log=True),
        "grl_lambda_max":    trial.suggest_categorical("grl_lambda_max", [0.8, 1.0]),
        "lmmd_lambda_max":   trial.suggest_float("lmmd_lambda_max", 0.25, 0.7, log=True),
        "lmmd_t":            trial.suggest_categorical("lmmd_t", [0.5, 0.8, 1.0, 1.2, 1.5]),
        "pseudo_thresh":     trial.suggest_categorical("pseudo_thresh", [0.85, 0.9, 0.95]),
    }

def objective_lmmd_only(trial,
                        source_train='../datasets/source/train/DC_T197_RP.txt',
                        source_val  ='../datasets/source/test/DC_T197_RP.txt',
                        target_train='../datasets/target/train/HC_T185_RP.txt',
                        out_dir     ='../datasets/info_optuna_lmmd'):
    with open("../configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)["DANN_LMMD_INFO"]
    num_layers = cfg["num_layers"]; ksz = cfg["kernel_size"]; sc = cfg["start_channels"]
    num_epochs = cfg["num_epochs"]

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = suggest_lmmd_only(trial)

    # data
    src_tr, tgt_tr = get_dataloaders(source_train, target_train, batch_size=p["batch_size"])
    src_va = DataLoader(PKLDataset(source_val), batch_size=p["batch_size"], shuffle=False)
    ps_loader = get_pseudo_dataloaders(target_path=target_train, batch_size=p["batch_size"])

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
    c_cls = nn.CrossEntropyLoss(); c_dom = nn.CrossEntropyLoss()

    # train
    best_val = train_dann_lmmd(
        model, src_tr,src_va, tgt_tr,ps_loader,
        optimizer, c_cls, c_dom, device,
        num_epochs=num_epochs, num_classes=10,batchsize=p["batch_size"],
        pseudo_thresh=p["pseudo_thresh"],
        scheduler=scheduler,
        grl_lambda_max = p["grl_lambda_max"],
        lmmd_start_epoch=5,
        lmmd_t=p["lmmd_t"],
        lmmd_lambda_max=p["lmmd_lambda_max"]
    )

    # 评估：InfoMax + 域 gap
    metrics = infomax_unsup_score_from_loader(model, src_tr, tgt_tr, device,
                                              T=1.0, marg_weight=1.0)
    trial.set_user_attr("best_val", best_val)
    trial.set_user_attr("score", metrics["score"])
    trial.set_user_attr("domain_gap", metrics["domain_gap"])
    print(f"[InfoMax] score={metrics['score']:.4f} | gap={metrics['domain_gap']:.3f}")

    return float(metrics["score"]), float(metrics["domain_gap"])



if __name__ == "__main__":
    sampler = optuna.samplers.NSGAIISampler()
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    study.optimize(lambda t: objective_lmmd_only(
        t,
        source_train='../datasets/source/train/DC_T197_RP.txt',
        source_val  ='../datasets/source/test/DC_T197_RP.txt',
        target_train='../datasets/target/train/HC_T185_RP.txt',
        out_dir     ='../datasets/info_optuna_lmmd',
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
    os.makedirs('../datasets/info_optuna_lmmd', exist_ok=True)
    with open(os.path.join('../datasets/info_optuna_lmmd', "best_lmmd_only.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("[DANN+LMMD] Best:", best)
