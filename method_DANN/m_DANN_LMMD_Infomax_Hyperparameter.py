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
from models.get_no_label_dataloader import get_dataloaders,get_pseudo_dataloaders
import yaml
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_with_stats
from models.MMD import classwise_mmd_biased_weighted,suggest_mmd_gammas,infomax_loss_from_logits
from collections import deque


def adam_param_groups(model, weight_decay):
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
                            source_loader, target_loader,
                            optimizer, criterion_cls, criterion_domain,
                            device, num_epochs=20, num_classes=10,
                            pseudo_thresh=0.95,
                            scheduler=None, batch_size=16,
                            # InfoMax
                            im_T=1.0, im_weight=0.5, im_marg_w=1.0,
                            # Gating
                            grl_lambda_max=1,max_lambda = 0.6,
                            lmmd_start_epoch=5,lmmd_t=1
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
    plateau_best_state = None
    patience = 0

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
                model, target_loader, device, threshold=pseudo_thresh, T=lmmd_t
            )
            kept, total = stats["kept"], stats["total"]
            cov, margin_mean = stats["coverage"], stats["margin_mean"]
            if kept > 0:
                pl_ds = TensorDataset(pseudo_x, pseudo_y, pseudo_w)
                pl_loader = DataLoader(pl_ds, batch_size=batch_size, shuffle=True,drop_last=True)

        # 2) Gated LMMD weights
        lambda_mmd_eff = mmd_lambda(epoch, num_epochs, max_lambda=max_lambda, start_epoch=lmmd_start_epoch)

        # 3) epoch training
        model.train()
        it_src = iter(source_loader)
        it_tgt = iter(target_loader)
        it_pl  = iter(pl_loader) if pl_loader is not None else None
        len_src, len_tgt = len(source_loader), len(target_loader)
        len_pl = len(pl_loader) if pl_loader is not None else 0
        num_iters = max(len_src, len_tgt, len_pl) if len_pl > 0 else max(len_src, len_tgt)

        cls_loss_sum = dom_loss_sum = mmd_loss_sum = im_loss_sum = 0.0
        tot_loss_sum = 0.0
        tot_target_samples=tot_cls_samples = tot_dom_samples = 0
        dom_correct_src = dom_correct_tgt = 0
        dom_total_src = dom_total_tgt = 0

        for _ in range(num_iters):
            try: src_x, src_y = next(it_src)
            except StopIteration:
                it_src = iter(source_loader); src_x, src_y = next(it_src)
            try: tgt_x = next(it_tgt)
            except StopIteration:
                it_tgt = iter(target_loader); tgt_x = next(it_tgt)
            if isinstance(tgt_x, (tuple, list)): tgt_x = tgt_x[0]
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

            # forword
            # Put λ only into GRL
            model.lambda_ = float(dann_lambda(epoch, num_epochs, max_lambda=grl_lambda_max))
            cls_out_src, dom_out_src, feat_src = model(src_x, grl=True)
            cls_out_tgt, dom_out_tgt, feat_tgt = model(tgt_x, grl=True)

            # 1) Source classification
            loss_cls = criterion_cls(cls_out_src, src_y)

            # 2) Domain confrontation
            dom_label_src = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            dom_label_tgt = torch.ones(tgt_x.size(0),  dtype=torch.long, device=device)
            loss_dom = (
                criterion_domain(dom_out_src, dom_label_src) * src_x.size(0)
                + criterion_domain(dom_out_tgt, dom_label_tgt) * tgt_x.size(0)
            ) / (src_x.size(0) + tgt_x.size(0))

            # 3) InfoMax
            loss_im, h_cond, h_marg = infomax_loss_from_logits(cls_out_tgt, T=im_T, marg_weight=im_marg_w)
            loss_im = im_weight * loss_im

            # 3) Class-conditional LMMD (weighted, quality-gated)
            if tgt_pl_x is not None and lambda_mmd_eff > 0:
                _, _, feat_source = model(src_x, grl=False)
                _, _, feat_tgt_pl = model(tgt_pl_x, grl=False)
                feat_source = F.normalize(feat_source, dim=1)
                feat_tgt_pl = F.normalize(feat_tgt_pl, dim=1)
                if cached_gammas is None:
                    cached_gammas = suggest_mmd_gammas(feat_source.detach(), feat_tgt_pl.detach())
                loss_lmmd = classwise_mmd_biased_weighted(
                    feat_source, src_y, feat_tgt_pl, tgt_pl_y, tgt_pl_w,
                    num_classes=num_classes, gammas=cached_gammas,
                    min_count_per_class=2
                )
                loss_lmmd = lambda_mmd_eff * loss_lmmd
            else:
                loss_lmmd = src_x.new_tensor(0.0)

            loss = loss_cls + loss_dom + loss_im + loss_lmmd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # ------ Statistics ------
            cls_loss_sum  += loss_cls.item() * src_x.size(0)
            dom_loss_sum  += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            im_loss_sum   += loss_im.item()  * (tgt_x.size(0))
            mmd_loss_sum  += loss_lmmd.item() * src_x.size(0)
            tot_loss_sum  += loss.item()     * (src_x.size(0) + tgt_x.size(0))
            tot_cls_samples += src_x.size(0)
            tot_dom_samples += (src_x.size(0) + tgt_x.size(0))
            tot_target_samples += tgt_x.size(0)

            dom_correct_src += (dom_out_src.argmax(1) == dom_label_src).sum().item()
            dom_total_src   += dom_label_src.size(0)
            dom_correct_tgt += (dom_out_tgt.argmax(1) == dom_label_tgt).sum().item()
            dom_total_tgt   += dom_label_tgt.size(0)

        # ---- Epoch Log ----
        avg_cls = cls_loss_sum / max(1, tot_cls_samples)
        avg_dom = dom_loss_sum / max(1, tot_dom_samples)
        avg_im  = im_loss_sum  / max(1, tot_target_samples)
        avg_mmd = mmd_loss_sum / max(1, tot_cls_samples)
        avg_tot = tot_loss_sum / max(1, tot_dom_samples)
        acc_src = dom_correct_src / max(1, dom_total_src)
        acc_tgt = dom_correct_tgt / max(1, dom_total_tgt)
        dom_acc = 0.5 * (acc_src + acc_tgt)
        gap = abs(dom_acc - 0.5)
        if scheduler is not None: scheduler.step()

        gap_hist.append(gap)

        score = gap + avg_im

        # Record the optimal model
        if epoch > 15:
            improved_global = (score < best_score - 1e-6)
            if improved_global:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

            gap_ok = (len(gap_hist) == W) and (sum(gap_hist) / W < GAP_TH)
            if gap_ok:
                improved_plateau = (score < plateau_best_score - 1e-6)
                if improved_plateau:
                    plateau_best_score = score
                    plateau_best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                print(f"[EARLY-STOP] patience {patience}/{PATIENCE} | gap_ok={gap_ok} | score={score:.4f}")
                if patience >= PATIENCE:
                    print("[EARLY-STOP] Stopping: stable alignment and no score improvement.")
                    # Prioritize backloading the "plateau optimal"; otherwise, backloading the "global optimal"
                    if plateau_best_state is not None:
                        model.load_state_dict(plateau_best_state)
                    elif best_state is not None:
                        model.load_state_dict(best_state)
                    break
            else:

                patience = 0

    if plateau_best_state is not None:
        model.load_state_dict(plateau_best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    return model



# Search only for exclusive super parameters for combat
def suggest_adv_only(trial):
    params = {
        "batch_size":    trial.suggest_categorical("batch_size", [32, 48, 64]),
        "learning_rate": trial.suggest_float("learning_rate",8e-5, 5e-4, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 8e-5, 3e-4, log=True),
        "grl_lambda_max":    trial.suggest_categorical("grl_lambda_max", [0.5]),
        "lmmd_lambda_max":   trial.suggest_float("lmmd_lambda_max", 0.35, 0.5, log=True),
        "lmmd_t":            trial.suggest_categorical("lmmd_t", [2]),
        "pseudo_thresh":     trial.suggest_categorical("pseudo_thresh", [0.95]),
        "im_weight":     trial.suggest_categorical("im_weight", [0.8]),
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

    # Domain indivisibility (the smaller the gap, the better)
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
                       dataset_configs=None,
                       out_dir     ='../datasets/info_optuna_dann',
                       n_repeats=2,
                       ):
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = 40
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = suggest_adv_only(trial)
    # Accumulators for averaging
    total_score = 0.0
    total_gap = 0.0
    n_eval = 0

    for dcfg in dataset_configs:
        for _ in np.arange(n_repeats):

           # data
            src_tr, tgt_tr = get_dataloaders(
                dcfg["source_train"], dcfg["target_train"],
                batch_size=p["batch_size"]
            )


            model = Flexible_DANN(num_layers=num_layers,
                                  start_channels=sc,
                                  kernel_size=ksz,
                                  num_classes=10,
                                  lambda_=1).to(device)

            optimizer = torch.optim.Adam( adam_param_groups(model, p["weight_decay"]), lr=p["learning_rate"], betas=(0.9, 0.999), eps=1e-8 )
            # Scheduler (Cosine; same as the number of training epochs)
            max_epochs = num_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=p["learning_rate"] * 0.1
            )
            c_cls = nn.CrossEntropyLoss()
            c_dom = nn.CrossEntropyLoss()


            model = train_dann_infomax_lmmd(model, src_tr, tgt_tr,
                        optimizer, c_cls, c_dom, device,
                        num_epochs=num_epochs,
                        pseudo_thresh=p["pseudo_thresh"],
                        scheduler=scheduler,batch_size=p["batch_size"],
                        # InfoMax Hyperparameters
                        im_T=p["im_T"], im_weight=p["im_weight"], im_marg_w=1.0,
                        grl_lambda_max=p["grl_lambda_max"],max_lambda=p["lmmd_lambda_max"],
                        lmmd_start_epoch=4,lmmd_t=p["lmmd_t"]
                                               )

            metrics = infomax_unsup_score_from_loader(model, src_tr, tgt_tr, device,
                                                      T=1.0, marg_weight=1.0)

           # Accumulate for averaging
            total_score += float(metrics["score"])
            total_gap += float(metrics["domain_gap"])
            n_eval += 1

        # Avoid division by zero
        mean_score = total_score / max(1, n_eval)
        mean_gap = total_gap / max(1, n_eval)

        trial.set_user_attr("score", mean_score)
        trial.set_user_attr("domain_gap", mean_gap)

    print(f"[InfoMax] score={metrics['score']:.4f} | "
          f"gap={metrics['domain_gap']:.3f}")

    # record trial
    rec = {
        "trial": trial.number, **p,
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

    # 2. Objective: Minimize the InfoMax score and minimize the domain_gap.
    return float(metrics["score"]), float(metrics["domain_gap"])


# ========= main =========
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Direct search for dedicated hyperparameters for combat
    sampler = optuna.samplers.NSGAIISampler()
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    study.optimize(lambda t: objective_adv_only(
        t,
        [{
            "target_train": "../datasets/HC_T191_RP.txt",
            "source_train": "../datasets/DC_T197_RP.txt",
        },
            {
                "target_train": "../datasets/HC_T194_RP.txt",
                "source_train": "../datasets/DC_T197_RP.txt",
            },
            {
                "target_train": "../datasets/HC_T185_RP.txt",
                "source_train": "../datasets/DC_T197_RP.txt",
            }],
        n_repeats=2,
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
