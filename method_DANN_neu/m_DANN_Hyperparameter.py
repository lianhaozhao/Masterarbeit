import os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from models.Flexible_DANN import Flexible_DANN
from PKLDataset import PKLDataset
from utils.general_train_and_test import general_test_model
from models.get_no_label_dataloader import get_dataloaders

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

def get_loaders(src_train, src_val, tgt_train, batch_size=32,
                num_workers=4, pin_memory=True):
    src_tr = DataLoader(PKLDataset(src_train), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=pin_memory)
    src_va = DataLoader(PKLDataset(src_val),   batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    # 目标域无标签：PKLDataset 若返回 (x,y) 也会忽略 y
    tgt_tr = DataLoader(PKLDataset(tgt_train), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=pin_memory)
    return src_tr, src_va, tgt_tr

# ========= DANN =========
def dann_lambda(epoch, num_epochs, max_lambda=1.0):
    p = epoch / max(1, num_epochs - 1)
    return (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) * max_lambda

def train_dann(model, source_loader, target_loader, val_loader,
               optimizer, scheduler, device,
               max_epochs=30, lambda_max=1.0,
               trial=None, early_patience=5):
    ce = nn.CrossEntropyLoss()
    best_val, best_state, wait = float('inf'), None, 0

    for epoch in range(max_epochs):
        model.train()
        for (src_x, src_y), tgt_x in zip(source_loader, target_loader):
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

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

            lam = dann_lambda(epoch, max_epochs, max_lambda=lambda_max)
            loss = loss_cls + lam * loss_dom

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
                cls_out_src, _ = model(xb)
                v = ce(cls_out_src, yb)
                v_total += xb.size(0); v_sum += v.item() * xb.size(0)
            val_loss = v_sum / max(1, v_total)

        if scheduler is not None: scheduler.step()

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

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
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64]),  # 你也可改成 [32, 48, 64]
        "learning_rate": trial.suggest_float("learning_rate", 5e-4, 1e-3, log=True),
        "weight_decay":  trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 3e-4, 5e-4]),
        "lambda_max":    trial.suggest_categorical("lambda_max", [0.5, 1.0]),
    }
    return params


def objective_adv_only(trial,
                       # —— 固定结构/数据路径/批量大小（只改这里）——
                       fixed_arch = dict(num_layers=5, kernel_size=7, start_channels=16,
                                         batch_size=32, cnn_act='leakrelu', num_classes=10),
                       source_train='../datasets/source/train/DC_T197_RP.txt',
                       source_val  ='../datasets/source/validation/DC_T197_RP.txt',
                       target_train='../datasets/target/train/HC_T185_RP.txt',
                       out_dir     ='../datasets/info_optuna_dann_adv',
                       device=None):
    os.makedirs(out_dir, exist_ok=True)
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    p = suggest_adv_only(trial)

    # 数据（batch 固定）
    src_tr, tgt_tr = get_dataloaders(source_train, target_train, batch_size=p["batch_size"])
    src_tr, src_va, tgt_tr = get_loaders(
        source_train, source_val, target_train, batch_size=p["batch_size"]
    )

    # 模型（结构固定）
    model = Flexible_DANN(num_layers=fixed_arch["num_layers"],
                          start_channels=fixed_arch["start_channels"],
                          kernel_size=fixed_arch["kernel_size"],
                          cnn_act=fixed_arch["cnn_act"],
                          num_classes=fixed_arch["num_classes"],
                          lambda_=1).to(device)

    # 优化器（Adam；BN/偏置不做 wd）
    optimizer = torch.optim.Adam(
        adam_param_groups(model, p["weight_decay"]),
        lr=p["learning_rate"], betas=(0.9, 0.999), eps=1e-8
    )

    # 调度器（Cosine；与训练轮数一致）
    max_epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=p["learning_rate"] * 0.1
    )

    best_val = train_dann(model, src_tr, tgt_tr, src_va, optimizer, scheduler, device,
                          max_epochs=max_epochs, lambda_max=p["lambda_max"],
                          trial=trial, early_patience=5)

    # 记录 trial
    rec = {"trial": trial.number, **p, "best_val_loss": float(best_val)}
    path = os.path.join(out_dir, "trials_adv_only.json")
    all_rec = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f: all_rec = json.load(f)
        except Exception: all_rec = []
    all_rec.append(rec)
    with open(path, "w") as f: json.dump(all_rec, f, indent=2)

    return best_val

# ========= 主程序 =========
if __name__ == "__main__":
    # set_seed(44)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 直接搜索对抗专属超参
    sampler = optuna.samplers.TPESampler(seed=44, n_startup_trials=10)
    pruner  = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    study.optimize(lambda t: objective_adv_only(
        t,
        fixed_arch=dict(num_layers=5, kernel_size=7, start_channels=16,
                        batch_size=32, cnn_act='leakrelu', num_classes=10),
        source_train='../datasets/source/train/DC_T197_RP.txt',
        source_val  ='../datasets/source/validation/DC_T197_RP.txt',
        target_train='../datasets/target/train/HC_T185_RP.txt',
        out_dir     ='../datasets/info_optuna_dann_adv',
        device=device
    ), n_trials=40)

    best = {
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_trial.params,
        "best_value": study.best_value
    }
    out_dir = '../datasets/info_optuna_dann_adv'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "best_adv_only.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("[DANN-AdvOnly] Best:", best)
