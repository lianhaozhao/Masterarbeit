import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import random
import yaml
from models.Flexible_DANN import Flexible_DANN
from PKLDataset import PKLDataset
from utils.general_train_and_test import general_test_model
from models.get_no_label_dataloader import get_dataloaders

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def adam_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #  norm weights & biases: no weight decay
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def dann_lambda(epoch, num_epochs,max_lambda = 1):
    """
    Common DANN λ schedule: smoothly increases from 0 to max_lambda

    """
    p = epoch / max(1, num_epochs - 1)
    return (2. / (1. + np.exp(-7 * p)) - 1.) * max_lambda

def train_dann(model, source_loader, target_loader,
               optimizer, criterion_cls, criterion_domain,
               device, num_epochs=20, dann_lambda=dann_lambda,scheduler = None,
               save_dir = None):
    best_gap = 0.5
    best_model_state = None
    patience = 0
    # ==== Create save directory ====
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        first_path = os.path.join(save_dir, f"first.pth")
        final_path = os.path.join(save_dir, f"final.pth")
    else:
        first_path = final_path = None
    for epoch in range(num_epochs):
        cls_loss_sum, dom_loss_sum, total_loss_sum = 0.0, 0.0, 0.0
        total_cls_samples, total_dom_samples = 0, 0
        dom_correct, dom_total = 0, 0
        model.train()

        for (src_x, src_y), tgt_x in zip(source_loader, target_loader):
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            model.lambda_ = float(dann_lambda(epoch, num_epochs,max_lambda = 1))
            cls_out_src, dom_out_src = model(src_x)
            _, dom_out_tgt = model(tgt_x)
            loss_cls = criterion_cls(cls_out_src, src_y)

            # Domain classification loss（DANN）
            dom_label_src = torch.zeros(src_x.size(0), dtype=torch.long, device=device)
            dom_label_tgt = torch.ones(tgt_x.size(0), dtype=torch.long, device=device)
            bs_src, bs_tgt = src_x.size(0), tgt_x.size(0)
            loss_dom_src = criterion_domain(dom_out_src, dom_label_src)
            loss_dom_tgt = criterion_domain(dom_out_tgt, dom_label_tgt)

            # Sample size-weighted "single-domain loss mean"
            loss_dom = (loss_dom_src * bs_src + loss_dom_tgt * bs_tgt) / (bs_src + bs_tgt)

            dom_preds_src = torch.argmax(dom_out_src, dim=1)
            dom_preds_tgt = torch.argmax(dom_out_tgt, dim=1)
            dom_correct += (dom_preds_src == dom_label_src).sum().item()
            dom_correct += (dom_preds_tgt == dom_label_tgt).sum().item()
            dom_total += dom_label_src.size(0) + dom_label_tgt.size(0)
            loss = loss_cls + loss_dom

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            cls_loss_sum += loss_cls.item() * src_x.size(0)
            dom_loss_sum += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            total_loss_sum += loss.item() * (src_x.size(0) + tgt_x.size(0))

            total_cls_samples += src_x.size(0)
            total_dom_samples += (src_x.size(0) + tgt_x.size(0))

        avg_cls_loss = cls_loss_sum / total_cls_samples
        avg_dom_loss = dom_loss_sum / total_dom_samples
        avg_total_loss = total_loss_sum / total_dom_samples

        # Domain classification accuracy (full round)
        dom_acc = dom_correct / dom_total
        gap = abs(dom_acc - 0.5)

        if scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch + 1}] Total Loss: {avg_total_loss:.4f} | "
              f"Cls loss: {avg_cls_loss:.4f} | Dom loss: {avg_dom_loss:.4f} | "
              f"DomAcc: {dom_acc:.4f}")
        if epoch == 0 and first_path is not None:
            torch.save(model.state_dict(), first_path)
            print(f"[SAVE] First epoch model saved: {first_path}")


        if gap < 0.05 and avg_cls_loss < 0.05 and epoch > 10:
            if gap < best_gap:
                best_gap = gap
                best_model_state = copy.deepcopy(model.state_dict())
            print(f"[INFO] patience {patience} / 3")
            patience += 1
            if patience > 3:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                print("[INFO] Early stopping: domain aligned and classifier converged.")
                break
        else:
            patience = 0
            best_gap = gap

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if final_path is not None:
        torch.save(model.state_dict(), final_path)
        print(f"[SAVE] Final model saved: {final_path}")

    return model


if __name__ == '__main__':
    # set_seed(seed=44)
    with open("../configs/default.yaml", 'r') as f:
        config = yaml.safe_load(f)['DANN_LMMD_INFO']
    batch_size = config['batch_size']
    learning_rate = 0.0011149529810769747
    weight_decay = 3.3975734111741645e-06
    num_layers = config['num_layers']
    kernel_size = config['kernel_size']
    start_channels = config['start_channels']
    num_epochs = 30

    files = [185, 188, 191, 194, 197]
    for file in files:

        source_path= '../datasets/DC_T197_RP.txt'
        target_path = '../datasets/HC_T{}_RP.txt'.format(file)
        target_test_path = '../datasets/HC_T{}_RP.txt'.format(file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading HC_T{file} ...")

        for run_id in range(10):

            model = Flexible_DANN(num_layers=num_layers,
                                  start_channels=start_channels,
                                  kernel_size=kernel_size,
                                  cnn_act='leakrelu',
                                  num_classes=10,
                                  lambda_=1).to(device)

            source_loader, target_loader = get_dataloaders(source_path, target_path, batch_size)

            optimizer = torch.optim.Adam(
                adam_param_groups(model, weight_decay),
                lr=learning_rate, betas=(0.9, 0.999)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=learning_rate * 0.1
            )
            criterion_cls = nn.CrossEntropyLoss()
            criterion_domain = nn.CrossEntropyLoss()

            print("[INFO] Starting standard DANN training (no pseudo labels)...")
            model=train_dann(model, source_loader, target_loader,
                       optimizer, criterion_cls, criterion_domain,
                       device, num_epochs=num_epochs, dann_lambda=dann_lambda,scheduler=scheduler,
                       save_dir=f"/content/drive/MyDrive/Masterarbeit/DANN/Model/HC_T{file}_Run{run_id}")

            print("[INFO] Evaluating on target test set...")
            test_dataset = PKLDataset(target_path)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            general_test_model(model, criterion_cls, test_loader, device)

            del model, optimizer, scheduler, source_loader, target_loader, test_loader, test_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()