import copy, math, random, os
import yaml
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.Flexible_DANN_LMMD import Flexible_DANN
from PKLDataset import PKLDataset
from models.get_no_label_dataloader import get_dataloaders
from utils.general_train_and_test import general_test_model

def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# DANN Lambda Scheduling (GRL only)
def dann_lambda(epoch, num_epochs, max_lambda=0.5):
    p = epoch / max(1, num_epochs-1)
    return (2.0 / (1.0 + np.exp(-3 * p)) - 1.0) * max_lambda

# Baseline weight of MMD (multiplied by quality gate to get final weight)
def mmd_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=5):
    if epoch < start_epoch:
        return 0.0
    p = (epoch - start_epoch) / max(1, (num_epochs - 1 - start_epoch))
    return (2.0 / (1.0 + np.exp(-3 * p)) - 1.0) * max_lambda

def adam_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 对于一维参数（例如 LN/BN 的权重 γ、bias），不加 weight decay
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

#  Weighted Class Conditional MMD (Multi-core RBF)
def _pairwise_sq_dists(a, b):
    # a: [m,d], b: [n,d]
    a2 = (a*a).sum(dim=1, keepdim=True)       # [m,1]
    b2 = (b*b).sum(dim=1, keepdim=True).t()   # [1,n]
    return a2 + b2 - 2 * (a @ b.t())

def _mk_kernel(a, b, gammas):
    d2 = _pairwise_sq_dists(a, b).clamp_min(0)
    k = 0.0
    M = max(1, len(gammas))
    for g in gammas:
        k = k + torch.exp(-float(g) * d2)
    return k / M

def _weighted_mean_kernel(K, w_row, w_col):
    # E_w[k] = (w_row^T K w_col) / (sum(w_row)*sum(w_col))
    num = (w_row.view(1,-1) @ K @ w_col.view(-1,1)).squeeze()
    den = (w_row.sum() * w_col.sum()).clamp_min(1e-8)
    return num / den

def mmd2_weighted(a, b, w_a=None, w_b=None, gammas=(0.5,1,2,4,8)):
    # MMD^2 = E_aa k + E_bb k - 2 E_ab k  （weight）
    if w_a is None: w_a = torch.ones(a.size(0), device=a.device)
    if w_b is None: w_b = torch.ones(b.size(0), device=b.device)
    Kaa = _mk_kernel(a, a, gammas)
    Kbb = _mk_kernel(b, b, gammas)
    Kab = _mk_kernel(a, b, gammas)
    e_aa = _weighted_mean_kernel(Kaa, w_a, w_a)
    e_bb = _weighted_mean_kernel(Kbb, w_b, w_b)
    e_ab = _weighted_mean_kernel(Kab, w_a, w_b)
    return (e_aa + e_bb - 2*e_ab).clamp_min(0.0)
@torch.no_grad()
def suggest_mmd_gammas(x_src, x_tgt, scales=(0.25,0.5,1,2,4)):
    x = torch.cat([x_src.detach(), x_tgt.detach()], dim=0)

    xi, xj = x.unsqueeze(1), x.unsqueeze(0)
    d2 = (xi - xj).pow(2).sum(-1).flatten()

    m = d2.clamp_min(1e-12).median()
    g0 = (1.0 / (2.0 * m)).item()
    return [s * g0 for s in scales]

def mmd2_unconditional(feat_src, feat_tgt, gammas):
    return mmd2_weighted(feat_src, feat_tgt, w_a=None, w_b=None, gammas=gammas)

def train_dann_mmd(model,
                   source_loader, target_loader,
                   optimizer, criterion_cls, criterion_domain,
                   device, num_epochs=20,
                   scheduler=None,max_lambda=1,
                   mmd_start_epoch=5):
    W, GAP_TH, PATIENCE = 5, 0.05, 3
    gap_hist = deque(maxlen=W)
    best_score = float('inf'); best_state = None
    plateau_best_score = float('inf'); plateau_best_state = None
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        it_src = iter(source_loader)
        it_tgt = iter(target_loader)
        len_src, len_tgt = len(source_loader), len(target_loader)
        num_iters = max(len_src, len_tgt)

        cls_loss_sum = dom_loss_sum = mmd_loss_sum = 0.0
        tot_loss_sum = 0.0
        tot_target_samples = tot_cls_samples = tot_dom_samples = 0
        dom_correct_src = dom_correct_tgt = dom_total_src = dom_total_tgt = 0
        mmd_steps = 0
        cached_gammas = None

        for _ in range(num_iters):
            try: src_x, src_y = next(it_src)
            except StopIteration:
                it_src = iter(source_loader); src_x, src_y = next(it_src)
            try: tgt_x = next(it_tgt)
            except StopIteration:
                it_tgt = iter(target_loader); tgt_x = next(it_tgt)

            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            model.lambda_ = float(dann_lambda(epoch, num_epochs, max_lambda=max_lambda))
            cls_out_src, dom_out_src, _ = model(src_x, grl=True)
            _,           dom_out_tgt, _ = model(tgt_x, grl=True)

            loss_cls = criterion_cls(cls_out_src, src_y)
            dom_label_src = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            dom_label_tgt = torch.ones(tgt_x.size(0),  dtype=torch.long, device=device)
            loss_dom = (
                criterion_domain(dom_out_src, dom_label_src) * src_x.size(0)
                + criterion_domain(dom_out_tgt, dom_label_tgt) * tgt_x.size(0)
            ) / (src_x.size(0) + tgt_x.size(0))

            # ---- unconditional MMD (no pseudo labels) ----
            lambda_mmd = mmd_lambda(epoch, num_epochs, max_lambda=0.3437, start_epoch=mmd_start_epoch)
            _, _, feat_src_plain = model(src_x, grl=False)
            _, _, feat_tgt_plain = model(tgt_x, grl=False)
            fs = F.normalize(feat_src_plain, dim=1)
            ft = F.normalize(feat_tgt_plain, dim=1)
            if cached_gammas is None:
                cached_gammas = suggest_mmd_gammas(fs.detach(), ft.detach())  # 你已有的函数
            gammas = cached_gammas
            loss_mmd = mmd2_unconditional(fs, ft, gammas)

            loss = loss_cls + loss_dom + lambda_mmd * loss_mmd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # stats
            cls_loss_sum += loss_cls.item() * src_x.size(0)
            dom_loss_sum += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            mmd_loss_sum += (lambda_mmd * loss_mmd).item(); mmd_steps += 1
            tot_loss_sum += loss.item() * (src_x.size(0) + tgt_x.size(0))
            tot_cls_samples += src_x.size(0); tot_dom_samples += (src_x.size(0) + tgt_x.size(0))
            tot_target_samples += tgt_x.size(0)
            dom_correct_src += (dom_out_src.argmax(1) == dom_label_src).sum().item()
            dom_total_src   += dom_label_src.size(0)
            dom_correct_tgt += (dom_out_tgt.argmax(1) == dom_label_tgt).sum().item()
            dom_total_tgt   += dom_label_tgt.size(0)

        avg_cls = cls_loss_sum / max(1, tot_cls_samples)
        avg_dom = dom_loss_sum / max(1, tot_dom_samples)
        avg_mmd = mmd_loss_sum / max(1, mmd_steps)
        avg_tot = tot_loss_sum / max(1, tot_dom_samples)
        acc_src = dom_correct_src / max(1, dom_total_src)
        acc_tgt = dom_correct_tgt / max(1, dom_total_tgt)
        dom_acc = 0.5 * (acc_src + acc_tgt); gap = abs(dom_acc - 0.5)
        if scheduler is not None: scheduler.step()

        print(f"[Epoch {epoch+1}] Total:{avg_tot:.4f} | CE:{avg_cls:.4f} | Dom:{avg_dom:.4f} "
              f"| MMD:{avg_mmd:.4f} (steps:{mmd_steps}) | DomAcc:{dom_acc:.4f} | "
              f"λ_GRL:{model.lambda_:.4f} | λ_MMD:{lambda_mmd:.4f}")

        gap_hist.append(gap)
        score = gap
        if epoch > 10:
            if score < best_score - 1e-6:
                best_score = score; best_state = copy.deepcopy(model.state_dict())
            gap_ok = (len(gap_hist) == W) and (sum(gap_hist) / W < GAP_TH)
            if gap_ok:
                if score < plateau_best_score - 1e-6:
                    plateau_best_score = score; plateau_best_state = copy.deepcopy(model.state_dict()); patience = 0
                else:
                    patience += 1
                print(f"[EARLY-STOP] patience {patience}/{PATIENCE} | gap_ok={gap_ok} | score={score:.4f}")
                if patience >= PATIENCE:
                    print("[EARLY-STOP] Stopping.")
                    if plateau_best_state is not None:
                        model.load_state_dict(plateau_best_state)
                    elif best_state is not None:
                        model.load_state_dict(best_state)
                    break

    if plateau_best_state is not None:
        model.load_state_dict(plateau_best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    bs = cfg['batch_size']
    lr = 0.002659971797592706
    wd = 0.002181843882109395
    num_layers = cfg['num_layers']
    ksz = cfg['kernel_size']
    sc = cfg['start_channels']
    num_epochs = 10

    files = [185]
    # files = [185, 188, 191, 194, 197]
    for file in files:

        src_path = '../datasets/source/train/DC_T197_RP.txt'
        tgt_path = '../datasets/target/train/HC_T{}_RP.txt'.format(file)
        tgt_test = '../datasets/target/test/HC_T{}_RP.txt'.format(file)

        print(f"[INFO] Loading HC_T{file} ...")

        for run_id in range(5):
            print(f"\n========== RUN {run_id} ==========")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Flexible_DANN(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                  cnn_act='leakrelu', num_classes=10, lambda_=1.0).to(device)

            src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
            optimizer = torch.optim.Adam(
                adam_param_groups(model, wd),
                lr=lr, betas=(0.9, 0.999)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.1)
            c_cls = nn.CrossEntropyLoss(); c_dom = nn.CrossEntropyLoss()

            print("[INFO] Starting DANN +MMD ...")
            model = train_dann_mmd(
                model,src_loader, tgt_loader,
                optimizer, c_cls, c_dom, device,
                num_epochs=num_epochs,
                scheduler=scheduler, max_lambda=0.8,
                mmd_start_epoch=1,
            )

            print("[INFO] Evaluating on target test set...")
            test_ds = PKLDataset(tgt_test)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
            general_test_model(model, c_cls, test_loader, device)

            del model, optimizer, scheduler, src_loader, tgt_loader, test_loader, test_ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

