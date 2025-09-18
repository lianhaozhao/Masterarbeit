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
from models.get_no_label_dataloader import get_dataloaders, get_pseudo_dataloaders
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_with_stats
from utils.general_train_and_test import general_test_model
from models.MMD import classwise_mmd_biased_weighted,suggest_mmd_gammas, infomax_loss_from_logits

class MultiPrototypes(nn.Module):
    def __init__(self, num_classes, feat_dim, num_protos=3, momentum=0.95):
        super().__init__()
        self.num_classes = num_classes
        self.num_protos = num_protos
        self.m = float(momentum)
        # [C, K, d]
        self.register_buffer('proto', torch.zeros(num_classes, num_protos, feat_dim))

    @torch.no_grad()
    def update(self, feats, labels, weights=None):
        if feats.numel() == 0:
            return
        device = feats.device  # 保持和输入特征一致
        for c in labels.unique():
            mask = (labels == c)
            if mask.sum() == 0: continue
            vec = feats[mask].mean(dim=0)

            # 确保 proto 在同一个 device
            proto_c = self.proto[c].to(device)

            sims = F.cosine_similarity(proto_c, vec.unsqueeze(0), dim=1)  # [K]
            k = sims.argmax().item()

            # 更新时也要保持 device 一致
            self.proto[c, k] = (
                    self.m * self.proto[c, k].to(device) + (1 - self.m) * vec
            )

    def supcon_logits(self, feats, tau=0.1, agg="max"):
        """
        feats: [N, d]
        return: [N, C]
        """
        f = F.normalize(feats, dim=1, eps=1e-8)             # [N, d]
        p = F.normalize(self.proto.to(device), dim=2, eps=1e-8)       # [C, K, d]

        # [N, d] @ [C*K, d]^T -> [N, C*K]
        logits_all = f @ p.reshape(-1, p.size(-1)).t()
        logits_all = logits_all / float(tau)

        # reshape -> [N, C, K]
        logits_all = logits_all.view(feats.size(0), self.num_classes, self.num_protos)

        # 聚合
        if agg == "max":
            logits = logits_all.max(dim=2).values
        elif agg == "mean":
            logits = logits_all.mean(dim=2)
        elif agg == "lse":  # log-sum-exp
            logits = torch.logsumexp(logits_all, dim=2)
        else:
            raise ValueError(f"Unknown agg={agg}")
        return logits

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
                            source_loader, target_loader,ps_loader,
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
    proto_module = MultiPrototypes(num_classes=10, feat_dim=512, num_protos=4).to(device)


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
                pl_loader = DataLoader(pl_ds, batch_size=batch_size, shuffle=True,drop_last=True)

        # 2) Gated LMMD weights
        lambda_proto_eff = mmd_lambda(epoch, num_epochs, max_lambda=max_lambda, start_epoch=lmmd_start_epoch)

        # 3) epoch training
        model.train()
        it_src = iter(source_loader)
        it_tgt = iter(target_loader)
        it_pl  = iter(pl_loader) if pl_loader is not None else None
        len_src, len_tgt = len(source_loader), len(target_loader)
        len_pl = len(pl_loader) if pl_loader is not None else 0
        num_iters = max(len_src, len_tgt, len_pl) if len_pl > 0 else max(len_src, len_tgt)

        cls_loss_sum = dom_loss_sum = proto_loss_sum = im_loss_sum = 0.0
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

            # 3)
            if tgt_pl_x is not None and lambda_proto_eff > 0:
                _, _, feat_tgt_pl = model(tgt_pl_x, grl=False)
                proto_module.update(feat_tgt_pl, tgt_pl_y)
                logits = proto_module.supcon_logits(feat_tgt_pl, tau=0.1, agg="mean")
                loss_proto = F.cross_entropy(logits, tgt_pl_y)
                loss_proto = lambda_proto_eff * loss_proto
            else:
                loss_proto = src_x.new_tensor(0.0)

            loss = loss_cls + loss_dom + loss_im + loss_proto

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # ------ Statistics ------
            cls_loss_sum  += loss_cls.item() * src_x.size(0)
            dom_loss_sum  += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            im_loss_sum   += loss_im.item()  * (tgt_x.size(0))
            proto_loss_sum  += loss_proto.item() * src_x.size(0)
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
        avg_proto = proto_loss_sum / max(1, tot_cls_samples)
        avg_tot = tot_loss_sum / max(1, tot_dom_samples)
        acc_src = dom_correct_src / max(1, dom_total_src)
        acc_tgt = dom_correct_tgt / max(1, dom_total_tgt)
        dom_acc = 0.5 * (acc_src + acc_tgt)
        gap = abs(dom_acc - 0.5)
        if scheduler is not None: scheduler.step()

        print(f"[Epoch {epoch+1}] Total loss:{avg_tot:.4f} | Avg cls loss:{avg_cls:.4f} | avg Dom loss:{avg_dom:.4f} "
              f"| avg IM loss:{avg_im:.4f} | avg proto loss:{avg_proto:.4f} | DomAcc:{dom_acc:.4f} | "
              f"cov:{cov:.2%} margin:{margin_mean:.3f} | "
              f"λ_GRL:{model.lambda_:.4f} | λ_proto_eff:{lambda_proto_eff:.4f}")

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
        # print("[INFO] Evaluating on target test set...")
        # target_test_path = '/content/datasets/target/test/HC_T185_RP.txt'
        # test_dataset = PKLDataset(target_test_path)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # general_test_model(model, criterion_cls, test_loader, device)

    if plateau_best_state is not None:
        model.load_state_dict(plateau_best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    return model


if __name__ == "__main__":
    with open("/content/github/configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    bs = 64
    lr = 0.0004490552594198228
    wd = 0.00023044104330072158
    num_layers = cfg['num_layers']
    ksz = cfg['kernel_size']
    sc = cfg['start_channels']
    num_epochs = cfg['num_epochs']

    # files = [185]
    files = [185, 188, 191, 194, 197]
    for file in files:

        src_path = '/content/datasets/source/train/DC_T197_RP.txt'
        tgt_path = '/content/datasets/target/train/HC_T{}_RP.txt'.format(file)
        tgt_test = '/content/datasets/target/test/HC_T{}_RP.txt'.format(file)

        print(f"[INFO] Loading HC_T{file} ...")

        for run_id in range(10):
            print(f"\n========== RUN {run_id} ==========")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Flexible_DANN(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                  cnn_act='leakrelu', num_classes=10, lambda_=1.0).to(device)

            src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
            ps_loader = get_pseudo_dataloaders(tgt_path, batch_size=bs)
            optimizer = torch.optim.Adam(
                adam_param_groups(model, wd),
                lr=lr, betas=(0.9, 0.999)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.1)
            c_cls = nn.CrossEntropyLoss(); c_dom = nn.CrossEntropyLoss()

            print("[INFO] Starting DANN + InfoMax + (quality-gated) LMMD ...")
            model = train_dann_infomax_lmmd(
                model, src_loader, tgt_loader,ps_loader,
                optimizer, c_cls, c_dom, device,
                num_epochs=num_epochs, num_classes=10,
                pseudo_thresh=0.95,
                scheduler=scheduler, batch_size=bs,
                # InfoMax Hyperparameters
                im_T=1, im_weight=0.8, im_marg_w=1.0,
                grl_lambda_max=0.5, max_lambda=0.35,
                lmmd_start_epoch=4, lmmd_t=1.5
            )

            print("[INFO] Evaluating on target test set...")
            test_ds = PKLDataset(tgt_test)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
            general_test_model(model, c_cls, test_loader, device)

            del model, optimizer, scheduler, src_loader, tgt_loader, test_loader, test_ds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()