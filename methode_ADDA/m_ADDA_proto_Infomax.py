import copy, math, random, os
import yaml
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.Flexible_ADDA import Flexible_ADDA, freeze, unfreeze, DomainClassifier
from PKLDataset import PKLDataset
from models.get_no_label_dataloader import get_dataloaders
from utils.general_train_and_test import general_test_model
from models.MMD import infomax_loss_from_logits
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_with_stats

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
        device = feats.device  # Maintain consistency with input features
        for c in labels.unique():
            mask = (labels == c)
            if mask.sum() == 0: continue
            vec = feats[mask].mean(dim=0)

            # Ensure proto is on the same device
            proto_c = self.proto[c].to(device)

            sims = F.cosine_similarity(proto_c, vec.unsqueeze(0), dim=1)  # [K]
            k = sims.argmax().item()

            # Device consistency must be maintained during updates.
            self.proto[c, k] = (
                    self.m * self.proto[c, k].to(device) + (1 - self.m) * vec
            )

    def supcon_logits(self, feats, tau=0.1, agg="max"):
        """
        feats: [N, d]
        return: [N, C]
        """
        f = F.normalize(feats, dim=1, eps=1e-8)             # [N, d]
        p = F.normalize(self.proto, dim=2, eps=1e-8)       # [C, K, d]

        # [N, d] @ [C*K, d]^T -> [N, C*K]
        logits_all = f @ p.reshape(-1, p.size(-1)).t()
        logits_all = logits_all / float(tau)

        # reshape -> [N, C, K]
        logits_all = logits_all.view(feats.size(0), self.num_classes, self.num_protos)

        if agg == "max":
            logits = logits_all.max(dim=2).values
        elif agg == "mean":
            logits = logits_all.mean(dim=2)
        elif agg == "lse":  # log-sum-exp
            logits = torch.logsumexp(logits_all, dim=2)
        else:
            raise ValueError(f"Unknown agg={agg}")
        return logits

def adam_param_groups(named_params, weight_decay):
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


def p_lambda(epoch, num_epochs, max_lambda=1e-1, start_epoch=5):
    if epoch < start_epoch:
        return 0.0
    p = (epoch - start_epoch) / max(1, (num_epochs - 1 - start_epoch))
    return (2.0 / (1.0 + np.exp(-5 * p)) - 1.0) * max_lambda


@torch.no_grad()
def copy_encoder_params(src_model, tgt_model, device):
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
        save_path=None,
):
    """
    Source domain pre-training:
    - Use only cross-entropy (CE)
    - Update feature extractor + classification head
    - With early stopping mechanism
    """
    src_model.train()
    best_loss = float("inf")
    best_state = None
    patience = 0
    PATIENCE_LIMIT = 3

    for epoch in range(num_epochs):
        src_model.train()
        tot_loss = tot_n = 0.0

        # ===== Source Domain CE Training =====
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

        # ===== Early Stop Logic =====
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
        src_model.load_state_dict(best_state)

    return src_model


# Phase 2 – ADDA  + InfoMax +prototype
def train_adda_infomax_lmmd(
        src_model, tgt_model, source_loader, target_loader,
        device, num_epochs=20, num_classes=10, batch_size=16,
        # Discriminator/Optimizer
        lr_ft=1e-4, lr_d=1e-4, wd=0.0, d_steps=1, ft_steps=1,
        # InfoMax
        im_T=1.0, im_weight=0.5, im_marg_w=1.0,
        # Pseudo+LMMD
        lmmd_start_epoch=3, pseudo_thresh=0.95, T_lmmd=1.5, max_lambda=35e-2, path=None,
):
    # 1) Frozen source model (completely fixed F_s + C)
    src_model.eval()
    freeze(src_model)

    # 2) Freeze the classifier of the target model and train only its encoder.
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
        _, feat_s, feat_c  = src_model(xb_s)
        feat_dim = feat_s.size(1)
        feat_dim_c = feat_c.size(1)
    D = DomainClassifier(feature_dim=feat_dim).to(device)
    opt_d = torch.optim.AdamW(
        adam_param_groups(D.named_parameters(), wd),
        lr=lr_d
    )
    c_dom = nn.CrossEntropyLoss().to(device)
    proto_module = MultiPrototypes(num_classes=10, feat_dim=feat_dim_c, num_protos=4).to(device)

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
        lambda_proto_eff = p_lambda(epoch, num_epochs, max_lambda=max_lambda, start_epoch=lmmd_start_epoch)

        it_src, it_tgt = iter(source_loader), iter(target_loader)
        it_pl = iter(pl_loader) if pl_loader is not None else None
        len_src, len_tgt = len(source_loader), len(target_loader)
        len_pl = len(pl_loader) if pl_loader is not None else 0
        steps = max(len_src, len_tgt, len_pl) if len_pl > 0 else max(len_src, len_tgt)
        # New iterator
        it_tgt_ft = iter(target_loader)

        tgt_model.train()
        tgt_model.classifier.eval()
        D.train()

        # count
        d_loss_sum = g_loss_sum = im_loss_sum = proto_loss_sum = ft_loss_sum = 0.0
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

            # (A) train D: max log D(F_s(xs)) + log (1 - D(F_t(xt)))
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

                # record D acc
                with torch.no_grad():
                    pred = d_out.argmax(1)
                    d_acc = (pred == d_lab).float().mean().item()
                    d_acc_sum += d_acc
                    d_cnt += 1
                    d_loss_sum += loss_d.item()

            # (B) Training F_t: min cross-entropy(D(F_t(xt)), “source” label)
            D.eval()
            for p in D.parameters():
                p.requires_grad = False

            for _k in range(ft_steps):
                if _k == 0 and xt_last is not None:
                    xt_ft = xt_last.to(device)  # Reuse the last batch of D
                else:
                    try:
                        xt_ft = next(it_tgt_ft)
                    except StopIteration:
                        it_tgt_ft = iter(target_loader)
                        xt_ft = next(it_tgt_ft)
                xt_ft = xt_ft.to(device)
                logits_t, f_t, _  = tgt_model(xt_ft)
                fool_lab = torch.ones(f_t.size(0), dtype=torch.long, device=device)  #Let D predict "source" -1
                g_out = D(f_t)
                loss_g = c_dom(g_out, fool_lab)

                # InfoMax
                loss_im, h_cond, h_marg = infomax_loss_from_logits(logits_t, T=im_T, marg_weight=im_marg_w)
                loss_im = im_weight * loss_im

                # LMMD：Use source tags and target pseudo tags for class conditional alignment.
                if it_pl is not None:
                    try:
                        xpl, ypl, wpl = next(it_pl)
                    except StopIteration:
                        it_pl = iter(pl_loader)
                        xpl, ypl, wpl = next(it_pl)
                    xpl, ypl, wpl = xpl.to(device), ypl.to(device), wpl.to(device)
                    _, _, feat_tgt  = tgt_model(xpl)
                    proto_module.update(feat_tgt, ypl)
                    logits = proto_module.supcon_logits(feat_tgt, tau=0.1, agg="mean")
                    loss_proto = F.cross_entropy(logits, ypl)
                    loss_proto = lambda_proto_eff * loss_proto
                else:
                    loss_proto = f_t.new_tensor(0.0)

                loss_ft = loss_g + loss_im + loss_proto
                opt_ft.zero_grad()
                loss_ft.backward()
                opt_ft.step()

                def to_scalar(x):
                    return x.detach().item() if torch.is_tensor(x) else float(x)

                g_loss_sum += to_scalar(loss_g)
                im_loss_sum += to_scalar(loss_im)
                proto_loss_sum += to_scalar(loss_proto)
                ft_loss_sum += to_scalar(loss_ft)
            for p in D.parameters():
                p.requires_grad = True
            D.train()

        # print
        print(
            f"[ADDA] Ep {epoch + 1}/{num_epochs} | "
            f"D:{d_loss_sum / max(1, steps * d_steps):.4f} | "
            f"G(adver):{g_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"IM:{im_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"proto:{proto_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"FT(total):{ft_loss_sum / max(1, steps * ft_steps):.4f} | "
            f"D-acc:{d_acc_sum / max(1, d_cnt):.4f} | "
            f"cov:{cov:.2%} margin:{margin_mean:.3f} | lambda_proto_eff:{float(lambda_proto_eff):.4f}"
        )
        scr = im_loss_sum / max(1, steps * ft_steps)
        if epoch > 10:
            if scr < best_loss:
                best_loss = scr
                best_state = copy.deepcopy(tgt_model.state_dict())


    if best_state is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(best_state, path)
        tgt_model.load_state_dict(best_state)

    return tgt_model


if __name__ == "__main__":
    with open("/content/github/configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    bs = 64
    lr_pre = 0.0009494768641358269
    wd_pre = 2.5e-4
    lr = 0.0002495284051956634
    wd = 6e-5

    num_layers = cfg['num_layers']
    ksz = cfg['kernel_size']
    sc = cfg['start_channels']
    num_epochs = 15
    pre_epochs = 15

    files = [185, 188, 191, 194, 197]
    for file in files:
        src_path = '/content/datasets/DC_T197_RP.txt'
        tgt_path = '/content/datasets/HC_T{}_RP.txt'.format(file)
        tgt_test = '/content/datasets/HC_T{}_RP.txt'.format(file)

        print(f"[INFO] Loading HC_T{file} ...")

        for run_id in range(10):
            print(f"\n========== RUN {run_id} (ADDA) ==========")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # —— Phase 1: Building and training the source domain model Fs + C
            src_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                      cnn_act='leakrelu', num_classes=10).to(device)

            src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
            optimizer_src = torch.optim.AdamW(
                adam_param_groups(src_model.named_parameters(), wd_pre),
                lr=lr_pre
            )
            scheduler_src = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_src, T_max=5, eta_min=lr_pre * 0.1)
            src_cls = nn.CrossEntropyLoss()

            print("[INFO] SRC pretrain (Fs + C) ...")
            src_model = pretrain_source_classifier(src_model, src_loader, optimizer_src, src_cls,
                                                   device,
                                                   num_epochs=pre_epochs, scheduler=scheduler_src,
                                                   save_path=f"/content/drive/MyDrive/Masterarbeit/ADDA_INFOMAX_PROTOTYPE/Model_Pre/HC_T{file}/RUN{run_id}.pth")

            # —— Phase 2: Initialize the target encoder Ft (copied from Fs), train ADDA + IM + prototype
            tgt_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                      cnn_act='leakrelu', num_classes=10).to(device)
            copy_encoder_params(src_model, tgt_model, device)

            print("[INFO] ADDA stage (Ft vs D) + InfoMax +prototype ...")
            tgt_model = train_adda_infomax_lmmd(
                src_model, tgt_model, src_loader, tgt_loader, device,
                num_epochs=num_epochs, num_classes=10, batch_size=bs,
                # Discriminator/Optimizer
                lr_ft=lr, lr_d=lr * 0.3, wd=wd, d_steps=1, ft_steps=1,
                # InfoMax
                im_T=1.0, im_weight=0.8, im_marg_w=1.0,
                # LMMD
                lmmd_start_epoch=3, pseudo_thresh=0.95, T_lmmd=1.5, max_lambda=0.5,
                path=f"/content/drive/MyDrive/Masterarbeit/ADDA_INFOMAX_PROTOTYPE/Model/HC_T{file}/RUN{run_id}.pth",

            )

            print("[INFO] Evaluating on target test set...")
            test_ds = PKLDataset(tgt_test)
            test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
            general_test_model(tgt_model, src_cls, test_loader, device)

            del src_model, tgt_model, optimizer_src, scheduler_src, src_loader, tgt_loader, test_loader, test_ds
            if torch.cuda.is_available(): torch.cuda.empty_cache()
