import os, json, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import optuna
import math
from models.Flexible_ADDA import Flexible_ADDA,freeze,unfreeze,DomainClassifier
from models.get_no_label_dataloader import get_dataloaders,get_pseudo_dataloaders
import yaml
from models.MMD import classwise_mmd_biased_weighted,suggest_mmd_gammas, infomax_loss_from_logits
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_with_stats


def adam_param_groups(named_params, weight_decay):
    """
    Build parameter groups for Adam/AdamW with and without weight decay.

    All parameters with 1D shape (e.g., LayerNorm/BatchNorm weights, biases)
    are put into a no-decay group; all others go to a decay group.

    Parameters
    ----------
    named_params : iterable of (str, Tensor)
        Iterable over (name, parameter) pairs, e.g. model.named_parameters().
    weight_decay : float
        Weight decay factor for parameters in the decay group.

    Returns
    -------
    list of dict
        Parameter groups compatible with torch.optim.Optimizer.
    """
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=5):
    """
    Smoothly schedule the MMD/LMMD weight from 0 to max_lambda.

    Uses a sigmoid-like schedule that is zero before start_epoch and
    grows to max_lambda towards the end of training.

    Parameters
    ----------
    epoch : int
        Current training epoch (0-based).
    num_epochs : int
        Total number of epochs.
    max_lambda : float, default 1e-1
        Maximum value of the scheduled weight.
    start_epoch : int, default 5
        Epoch at which MMD begins to turn on.

    Returns
    -------
    float
        Effective lambda for the current epoch.
    """
    if epoch < start_epoch:
        return 0.0
    p = (epoch - start_epoch) / max(1, (num_epochs - 1 - start_epoch))
    return (2.0 / (1.0 + np.exp(-3 * p)) - 1.0) * max_lambda


@torch.no_grad()
def copy_encoder_params(src_model, tgt_model, device):
    """
    Copy all parameters from src_model into tgt_model.

    This is used in ADDA to initialize the target encoder with
    the weights of the pre-trained source encoder + classifier.

    Parameters
    ----------
    src_model : nn.Module
        Source domain model (pre-trained).
    tgt_model : nn.Module
        Target domain model to be initialized.
    device : torch.device
        Device to move the target model to after copying.
    """
    tgt_model.load_state_dict(copy.deepcopy(src_model.state_dict()))
    tgt_model.to(device)


# Phase 1 – Training using only the source domain (F_s + C)
def pretrain_source_classifier(
        src_model,
        source_loader,
        optimizer,
        criterion_cls,
        device,
        num_epochs=5,
        scheduler=None,
):
    """
    Phase 1: Supervised pre-training on the source domain.

    - Uses only cross-entropy classification loss.
    - Updates both the feature extractor and the classifier head.
    - Includes a simple early stopping mechanism on training loss.

    Parameters
    ----------
    src_model : nn.Module
        Model to train on the source domain.
    source_loader : DataLoader
        Dataloader providing (x, y) from the source domain.
    optimizer : torch.optim.Optimizer
        Optimizer for the model.
    criterion_cls : callable
        Classification loss function (e.g., CrossEntropyLoss).
    device : torch.device
        Device to run training on.
    num_epochs : int, default 5
        Maximum number of training epochs.
    scheduler : optional
        Learning rate scheduler to step at the end of each epoch.

    Returns
    -------
    nn.Module
        The pre-trained model (restored to the best epoch).
    """
    src_model.train()
    best_loss = float("inf")
    best_state = None
    patience = 0
    PATIENCE_LIMIT = 3

    for epoch in range(num_epochs):
        src_model.train()
        tot_loss = tot_n = 0.0

        # ===== Source domain CE training =====
        for xb, yb in source_loader:
            xb, yb = xb.to(device), yb.to(device)

            logits_s, _, _ = src_model(xb)
            loss = criterion_cls(logits_s, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            tot_loss += loss.item() * bs
            tot_n += bs

        epoch_loss = tot_loss / max(1, tot_n)
        print(f"[SRC] Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f}")

        # ===== Early stopping logic =====
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = copy.deepcopy(src_model.state_dict())
            patience = 0
        else:
            patience += 1

        if patience >= PATIENCE_LIMIT:
            print(f"[EARLY STOP] No improvement for {PATIENCE_LIMIT} epochs, stop training.")
            break

        # ===== Learning rate scheduling =====
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        src_model.load_state_dict(best_state)

    return src_model


# Phase 2 —— ADDA  + InfoMax + LMMD
def train_adda_infomax_lmmd(
        src_model, tgt_model, source_loader, target_loader,
        device, num_epochs=20, num_classes=10, batch_size=16,
        # Discriminator/Optimizer
        lr_ft=1e-4, lr_d=1e-4, wd=0.0, d_steps=1, ft_steps=1,
        # InfoMax
        im_T=1.0, im_weight=0.5, im_marg_w=1.0,
        # Pseudo+LMMD
        lmmd_start_epoch=3, pseudo_thresh=0.95, T_lmmd=2, max_lambda=35e-2,
):
    """
    Phase 2: ADDA-style adversarial adaptation with InfoMax and LMMD.

    Procedure:
    ----------
    1) Freeze the source model completely (F_s + C_s).
    2) Initialize the target encoder from the source model, and freeze the
       target classifier head; only the target encoder + reducer are trained.
    3) Train a domain discriminator D to distinguish F_s(xs) (source) from
       F_t(xt) (target).
    4) Train F_t to fool D (adversarial alignment), while adding:
       - InfoMax loss on target logits (confidence + balanced use of classes).
       - Class-wise LMMD loss using target pseudo-labels.

    Parameters
    ----------
    src_model : nn.Module
        Frozen source model (encoder + classifier).
    tgt_model : nn.Module
        Target model to be adapted.
    source_loader : DataLoader
        Source domain dataloader (labeled).
    target_loader : DataLoader
        Target domain dataloader (unlabeled).
    device : torch.device
        Device for training.
    num_epochs : int, default 20
        Number of epochs for this adaptation phase.
    num_classes : int, default 10
        Number of classes for LMMD.
    batch_size : int, default 16
        Batch size for pseudo-label dataloader.

    lr_ft : float, default 1e-4
        Learning rate for the target encoder/reducer (feature transformer).
    lr_d : float, default 1e-4
        Learning rate for the domain discriminator D.
    wd : float, default 0.0
        Weight decay for both optimizers.
    d_steps : int, default 1
        Number of discriminator update steps per outer iteration.
    ft_steps : int, default 1
        Number of target encoder update steps per outer iteration.

    im_T : float, default 1.0
        Temperature for InfoMax softmax.
    im_weight : float, default 0.5
        Weight on the InfoMax loss.
    im_marg_w : float, default 1.0
        Marginal entropy weight inside the InfoMax loss.

    lmmd_start_epoch : int, default 3
        Epoch from which to start using LMMD and pseudo labels.
    pseudo_thresh : float, default 0.95
        Confidence threshold for selecting target pseudo labels.
    T_lmmd : float, default 2
        Temperature when computing pseudo labels for LMMD.
    max_lambda : float, default 0.35
        Maximum LMMD weight (combined with mmd_lambda schedule).

    Returns
    -------
    tgt_model : nn.Module
        Adapted target model (restored to the best epoch according to InfoMax).
    D : nn.Module
        Trained domain discriminator.
    """
    # 1) Frozen source model (completely fixed F_s + C)
    src_model.eval()
    freeze(src_model)

    # 2) Freeze the classifier of the target model and train only its encoder/reducer.
    for p in tgt_model.classifier.parameters():
        p.requires_grad = False
    enc_named_params = []
    for n, p in tgt_model.named_parameters():
        if ("feature_extractor" in n) or ("feature_reducer" in n):
            enc_named_params.append((n, p))

    opt_ft = torch.optim.AdamW(
        adam_param_groups(enc_named_params, wd),
        lr=lr_ft
    )

    # 3) Infer feature dimensions from a batch and construct D on demand.
    with torch.no_grad():
        xb_s, yb_s = next(iter(source_loader))
        xb_s = xb_s.to(device)
        _, f_s, _ = src_model(xb_s)
        feat_dim = f_s.size(1)
    D = DomainClassifier(feature_dim=feat_dim).to(device)
    opt_d = torch.optim.AdamW(
        adam_param_groups(D.named_parameters(), wd),
        lr=lr_d
    )
    c_dom = nn.CrossEntropyLoss().to(device)

    best_loss = float("inf")
    best_state = None

    # 4) Training loop (alternating optimization of D and F_t)
    for epoch in range(num_epochs):
        # Prepare pseudo-labels for LMMD
        pl_loader = None
        cached_gammas = None
        pseudo_x = pseudo_y = pseudo_w = None
        cov = margin_mean = 0.0
        if epoch >= lmmd_start_epoch:
            pseudo_x, pseudo_y, pseudo_w, stats = generate_pseudo_with_stats(
                tgt_model, target_loader, device, threshold=pseudo_thresh, T=T_lmmd
            )
            cov, margin_mean = stats["coverage"], stats["margin_mean"]
            if pseudo_x.numel() > 0:
                pl_ds = TensorDataset(pseudo_x, pseudo_y, pseudo_w)
                pl_loader = DataLoader(pl_ds, batch_size=batch_size, shuffle=True)
        lambda_mmd_eff = mmd_lambda(epoch, num_epochs, max_lambda=max_lambda, start_epoch=lmmd_start_epoch)

        it_src, it_tgt = iter(source_loader), iter(target_loader)
        it_pl = iter(pl_loader) if pl_loader is not None else None
        len_src, len_tgt = len(source_loader), len(target_loader)
        len_pl = len(pl_loader) if pl_loader is not None else 0
        steps = max(len_src, len_tgt, len_pl) if len_pl > 0 else max(len_src, len_tgt)
        it_tgt_ft = iter(target_loader)

        tgt_model.train()
        tgt_model.classifier.eval()
        D.train()

        # Accumulators for logging
        d_loss_sum = g_loss_sum = im_loss_sum = mmd_loss_sum = ft_loss_sum = 0.0
        d_acc_sum = 0.0
        d_cnt = 0

        for _ in range(steps):
            try:
                xs, ys = next(it_src)
            except StopIteration:
                it_src = iter(source_loader)
                xs, ys = next(it_src)
            try:
                xt = next(it_tgt)
            except StopIteration:
                it_tgt = iter(target_loader)
                xt = next(it_tgt)
            if isinstance(xt, (tuple, list)): xt = xt[0]
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            # (A) Train D: max log D(F_s(xs)) + log (1 - D(F_t(xt)))
            for _k in range(d_steps):
                with torch.no_grad():
                    _, f_s, _  = src_model(xs)  # [B, d]
                    _, f_t, _  = tgt_model(xt)  # [B, d]
                d_in = torch.cat([f_s, f_t], dim=0)
                d_lab = torch.cat([torch.ones(f_s.size(0)), torch.zeros(f_t.size(0))], dim=0).long().to(
                    device)  # 1=source,0=target
                d_out = D(d_in)
                loss_d = c_dom(d_out, d_lab)
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                xt_last = xt.detach()

                # Record D accuracy
                with torch.no_grad():
                    pred = d_out.argmax(1)
                    d_acc = (pred == d_lab).float().mean().item()
                    d_acc_sum += d_acc
                    d_cnt += 1
                    d_loss_sum += loss_d.item()

            # (B) Train F_t: min cross-entropy(D(F_t(xt)), "source" label)
            D.eval()
            for p in D.parameters():
                p.requires_grad = False

            for _k in range(ft_steps):
                if _k == 0 and xt_last is not None:
                    xt_ft = xt_last.to(device)  # Reuse the last batch from D
                else:
                    try:
                        xt_ft = next(it_tgt_ft)
                    except StopIteration:
                        it_tgt_ft = iter(target_loader)
                        xt_ft = next(it_tgt_ft)
                xt_ft = xt_ft.to(device)
                logits_t, f_t, _  = tgt_model(xt_ft)
                fool_lab = torch.ones(f_t.size(0), dtype=torch.long, device=device)  # encourage D to predict "source"
                g_out = D(f_t)
                loss_g = c_dom(g_out, fool_lab)

                # InfoMax
                loss_im, h_cond, h_marg = infomax_loss_from_logits(logits_t, T=im_T, marg_weight=im_marg_w)
                loss_im = im_weight * loss_im

                # LMMD: use source labels and target pseudo labels for class-conditional alignment.
                if it_pl is not None:
                    try:
                        xpl, ypl, wpl = next(it_pl)
                    except StopIteration:
                        it_pl = iter(pl_loader)
                        xpl, ypl, wpl = next(it_pl)
                    xpl, ypl, wpl = xpl.to(device), ypl.to(device), wpl.to(device)
                    _, _, f_s_n = src_model(xs)
                    f_s_n = F.normalize(f_s_n, dim=1)
                    _, _,f_t_pl = tgt_model(xpl)
                    f_t_pl_n = F.normalize(f_t_pl, dim=1)
                    if cached_gammas is None:
                        cached_gammas = suggest_mmd_gammas(f_s_n.detach(), f_t_pl_n.detach())
                    loss_lmmd = classwise_mmd_biased_weighted(
                        f_s_n, ys, f_t_pl_n, ypl, wpl,
                        num_classes=num_classes, gammas=cached_gammas, min_count_per_class=2
                    )
                    loss_lmmd = lambda_mmd_eff * loss_lmmd
                else:
                    loss_lmmd = f_t.new_tensor(0.0)

                loss_ft = loss_g + loss_im + loss_lmmd
                opt_ft.zero_grad()
                loss_ft.backward()
                opt_ft.step()

                def to_scalar(x):
                    return x.detach().item() if torch.is_tensor(x) else float(x)

                g_loss_sum += to_scalar(loss_g)
                im_loss_sum += to_scalar(loss_im)
                mmd_loss_sum += to_scalar(loss_lmmd)
                ft_loss_sum += to_scalar(loss_ft)
            for p in D.parameters():
                p.requires_grad = True
            D.train()

        # Logging
        print(
            f"[ADDA] Ep {epoch + 1}/{num_epochs} | "
            f"D:{d_loss_sum / max(1, steps * d_steps):.4f} | "
            f"G(adver):{g_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"IM:{im_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"LMMD:{mmd_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"FT(total):{ft_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"D-acc:{d_acc_sum / max(1, d_cnt):.4f} | "
            f"cov:{cov:.2%} margin:{margin_mean:.3f} | lambda_mmd_eff:{float(lambda_mmd_eff):.4f}"
        )
        scr = im_loss_sum / max(1, steps * ft_steps)
        if epoch > num_epochs // 2:
            if scr < best_loss:
                best_loss = scr
                best_state = copy.deepcopy(tgt_model.state_dict())

        if best_state is not None:
            tgt_model.load_state_dict(best_state)

    return tgt_model,D


# Search only for exclusive hyperparameters for adversarial training
def suggest_adv_only(trial):
    """
    Suggest hyperparameters for adversarial-only (InfoMax+LMMD+ADDA) tuning.

    This function defines the Optuna search space for:
      - batch size
      - learning rates and weight decay (pre-training and adaptation)
      - InfoMax parameters (here mostly fixed)
      - LMMD-related parameters (max_lambda, pseudo threshold, temperature)

    Parameters
    ----------
    trial : optuna.trial.Trial
        Current Optuna trial object.

    Returns
    -------
    dict
        Dictionary of suggested hyperparameters.
    """
    params = {
        "batch_size":    trial.suggest_categorical("batch_size", [32, 48, 64]),
        "learning_rate_pre": trial.suggest_float("learning_rate_pre", 1e-4, 1e-3, log=True),
        "weight_decay_pre": trial.suggest_float("weight_decay_pre", 1e-5, 1e-3, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 5e-5, 1e-3, log=True),
        "im_weight":     trial.suggest_categorical("im_weight", [0.8]),
        "im_T":          trial.suggest_categorical("im_T", [1.0]),
        "max_lambda": trial.suggest_categorical("max_lambda", [0.35,0.4,0.5]),
        "pseudo_thresh": trial.suggest_categorical("pseudo_thresh", [0.90,0.95]),
        "T_lmmd": trial.suggest_categorical("T_lmmd", [1.5,2]),
    }
    return params


@torch.no_grad()
def infomax_unsup_score_from_loader(model, D,source_loader, target_loader, device,
                                    T=1.0, marg_weight=1.0, eps=1e-8):
    """
    Compute an unsupervised InfoMax-based score + domain-gap metric.

    The score is:
        mean_entropy(p(y|x)) - marg_weight * H(mean_y p(y|x))
    (lower is better: lower conditional entropy + high marginal entropy).

    The domain_gap is based on the accuracy of D in distinguishing
    source vs target features:
        gap = |domain_acc - 0.5|
    (smaller is better: 0.5 corresponds to perfectly indistinguishable domains).

    Parameters
    ----------
    model : nn.Module
        Target model used to produce logits/features for the target set.
    D : nn.Module
        Domain discriminator trained on features from model.
    source_loader : DataLoader
        Source dataloader (only inputs are used here).
    target_loader : DataLoader
        Target dataloader.
    device : torch.device
        Device for evaluation.
    T : float, default 1.0
        Temperature for softmax.
    marg_weight : float, default 1.0
        Weight on the marginal entropy term.
    eps : float, default 1e-8
        Small constant for numerical stability.

    Returns
    -------
    dict
        Dictionary with:
        - "score": InfoMax unsupervised score (float)
        - "domain_gap": |domain_acc - 0.5| (float)
    """
    model.eval()
    probs = []
    for xb in target_loader:
        xb = xb.to(device)
        cls_out, _, _ = model(xb)
        p = F.softmax(cls_out / T, dim=1)
        probs.append(p.detach().cpu())
    if not probs:
        return {"score": float("inf"), "domain_gap": 0.5}
    P = torch.cat(probs, dim=0).clamp_min(eps)
    mean_entropy = (-P * P.log()).sum(dim=1).mean().item()
    p_bar = P.mean(dim=0).clamp_min(eps)
    marginal_entropy = (-(p_bar * p_bar.log()).sum()).item()
    score = mean_entropy - marg_weight * marginal_entropy

    # Domain indistinguishability (the smaller the gap, the better)
    tot, correct = 0, 0
    for xs, _ in source_loader:
        xs = xs.to(device)
        _, f_s, _ = model(xs)
        pred_s = D(f_s).argmax(dim=1)
        correct += (pred_s == 1).sum().item()
        tot += xs.size(0)
    for xt in target_loader:
        xt = xt.to(device)
        _, f_t, _ = model(xt)
        pred_t = D(f_t).argmax(dim=1)
        correct += (pred_t == 0).sum().item()
        tot += xt.size(0)
    domain_acc = correct / max(1, tot)
    domain_gap = abs(domain_acc - 0.5)

    return {"score": float(score), "domain_gap": float(domain_gap)}


def objective_adv_only(trial,
                       dataset_configs=None,
                       out_dir=None,
                       n_repeats=2,
                       ):
    """
    Optuna objective for adversarial-only (ADDA + InfoMax + LMMD) tuning.

    For each trial:
      - Sample hyperparameters (batch size, learning rates, etc.).
      - For each dataset configuration and repetition:
          * Pre-train a source classifier.
          * Initialize a target model from the source.
          * Run ADDA + InfoMax + LMMD adaptation.
          * Compute InfoMax unsupervised score and domain gap.
      - Average the metrics over all dataset/repeat combinations.

    The objective returns three values to be minimized:
      (mean_score, mean_domain_gap, mean_pretrain_loss)

    Parameters
    ----------
    trial : optuna.trial.Trial
        Current Optuna trial.
    dataset_configs : list of dict
        Each dict must contain "source_train" and "target_train" paths.
    out_dir : str
        Output directory where trial records will be stored.
    n_repeats : int, default 2
        Number of repetitions per dataset configuration.

    Returns
    -------
    tuple
        (mean_score, mean_domain_gap, mean_loss)
    """
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    num_layers = cfg['num_layers']
    ksz = cfg['kernel_size']
    sc = cfg['start_channels']
    num_epochs = 1
    source_epoch = 1
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = suggest_adv_only(trial)

    # Accumulators for averaging
    total_score = 0.0
    total_gap = 0.0
    total_loss = 0.0
    n_eval = 0

    for dcfg in dataset_configs:
        for _ in np.arange(n_repeats):
            # Data (fixed batch)
            src_loader, tgt_loader = get_dataloaders(
                dcfg["source_train"], dcfg["target_train"],
                batch_size=p["batch_size"]
            )

            # Model (fixed structure)
            src_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                      cnn_act='leakrelu', num_classes=10).to(device)
            optimizer_src = torch.optim.AdamW(
                adam_param_groups(src_model.named_parameters(), p["weight_decay_pre"]),
                lr=p["learning_rate_pre"], betas=(0.9, 0.999), eps=1e-8
            )
            scheduler_src = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_src, T_max=5, eta_min=p["learning_rate_pre"] * 0.1
            )
            src_cls = nn.CrossEntropyLoss()

            src_model = pretrain_source_classifier(
                src_model, src_loader, optimizer_src,
                src_cls, device,
                num_epochs=source_epoch, scheduler=scheduler_src
            )


            loss = 0.0  # if you want exact pretrain loss tracked, you can modify pretrain_source_classifier to return it

            tgt_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                      cnn_act='leakrelu', num_classes=10).to(device)
            copy_encoder_params(src_model, tgt_model, device)

            tgt_model, D = train_adda_infomax_lmmd(
                src_model, tgt_model, src_loader, tgt_loader, device,
                num_epochs=num_epochs, num_classes=10, batch_size=p['batch_size'],
                # Discriminator/Optimizer
                lr_ft=p["learning_rate"], lr_d=p["learning_rate"] * 0.5,
                wd=p["weight_decay"], d_steps=1, ft_steps=2,
                # InfoMax
                im_T=1.0, im_weight=0.8, im_marg_w=1.0,
                # Pseudo+LMMD
                lmmd_start_epoch=3, pseudo_thresh=p["pseudo_thresh"],
                T_lmmd=p["T_lmmd"], max_lambda=p["max_lambda"],
            )

            metrics = infomax_unsup_score_from_loader(
                tgt_model, D, src_loader, tgt_loader, device,
                T=1.0, marg_weight=1.0
            )

            # Accumulate for averaging
            total_score += float(metrics["score"])
            total_gap += float(metrics["domain_gap"])
            total_loss += float(loss)
            n_eval += 1

    # Avoid division by zero
    mean_score = total_score / max(1, n_eval)
    mean_gap = total_gap / max(1, n_eval)
    mean_loss = total_loss / max(1, n_eval)

    trial.set_user_attr("score", mean_score)
    trial.set_user_attr("domain_gap", mean_gap)
    trial.set_user_attr("loss", mean_loss)

    print(f"[InfoMax] mean_score={mean_score:.4f} | "
          f"mean_gap={mean_gap:.3f} | mean_loss={mean_loss:.4f}")

    # Record trial (store averages)
    rec = {
        "trial": trial.number, **p,
        "score": mean_score,
        "domain_gap": mean_gap,
        "loss": mean_loss,
    }
    path = os.path.join(out_dir, "trials_adv_only.json")
    all_rec = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                all_rec = json.load(f)
        except Exception:
            all_rec = []
    all_rec.append(rec)
    with open(path, "w") as f:
        json.dump(all_rec, f, indent=2)

    # Objective: now returns the averaged values
    return float(mean_score), float(mean_gap), float(mean_loss)


# ========= main =========
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Direct search for dedicated hyperparameters for adversarial adaptation
    sampler = optuna.samplers.NSGAIISampler()
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"], sampler=sampler)

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
        out_dir     ='../datasets/info_optuna_ADDA',
    ), n_trials=50)

    pareto = study.best_trials
    print("Pareto size:", len(pareto))
    for t in pareto[:5]:
        print("trial#", t.number, "values=", t.values, "params=", t.params)

    def scalarize(t,  beta=0.5):
        """
        Simple scalarization of the multi-objective values.

        score_scalar = score + beta * domain_gap + beta * loss
        (lower is better).
        """
        score, gap, loss = t.values
        return score + beta * gap + beta * loss

    chosen = min(pareto, key=lambda t: scalarize(t,  1))
    best = {
        "chosen_trial_number": chosen.number,
        "chosen_params": chosen.params,
        "chosen_values": chosen.values
    }
    out_dir = '../datasets/info_optuna_ADDA'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "best_adv_only.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("[ADDA-AdvOnly] Best:", best)
