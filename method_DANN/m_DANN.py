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
from models.get_no_label_dataloader import get_target_loader

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders(source_path, target_path, batch_size):
    source_dataset = PKLDataset(txt_path=source_path)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = get_target_loader(target_path, batch_size=batch_size, shuffle=True)
    return source_loader, target_loader

def dann_lambda(epoch, num_epochs):
    """
    常用的 DANN λ 调度：从 0 平滑升到 1
    你也可以把 -10 调轻/重来改变上升速度
    """
    p = epoch / float(num_epochs)
    return 2. / (1. + np.exp(-10 * p)) - 1.

def train_dann(model, source_loader, target_loader,
               optimizer, criterion_cls, criterion_domain,
               device, num_epochs=20, lambda_=0.1,scheduler = None):
    best_gap = 0.5
    best_model_state = None
    patience = 0
    for epoch in range(num_epochs):
        cls_loss_sum, dom_loss_sum, total_loss_sum = 0.0, 0.0, 0.0
        total_cls_samples, total_dom_samples = 0, 0
        dom_correct, dom_total = 0, 0
        model.train()

        for (src_x, src_y), tgt_x in zip(source_loader, target_loader):
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            cls_out_src, dom_out_src = model(src_x)
            _, dom_out_tgt = model(tgt_x)

            loss_cls = criterion_cls(cls_out_src, src_y)

            dom_label_src = torch.zeros(src_x.size(0), dtype=torch.long).to(device)
            dom_label_tgt = torch.ones(tgt_x.size(0), dtype=torch.long).to(device)
            loss_dom = criterion_domain(dom_out_src, dom_label_src) + \
                       criterion_domain(dom_out_tgt, dom_label_tgt)

            dom_preds_src = torch.argmax(dom_out_src, dim=1)
            dom_preds_tgt = torch.argmax(dom_out_tgt, dim=1)
            dom_correct += (dom_preds_src == dom_label_src).sum().item()
            dom_correct += (dom_preds_tgt == dom_label_tgt).sum().item()
            dom_total += dom_label_src.size(0) + dom_label_tgt.size(0)

            loss = loss_cls + lambda_ * loss_dom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cls_loss_sum += loss_cls.item() * src_x.size(0)
            dom_loss_sum += loss_dom.item() * (src_x.size(0) + tgt_x.size(0))
            total_loss_sum += loss.item() * (src_x.size(0) + tgt_x.size(0))

            total_cls_samples += src_x.size(0)
            total_dom_samples += (src_x.size(0) + tgt_x.size(0))

        avg_cls_loss = cls_loss_sum / total_cls_samples
        avg_dom_loss = dom_loss_sum / total_dom_samples
        avg_total_loss = total_loss_sum / total_dom_samples

        # 域分类准确率（整轮）
        dom_acc = dom_correct / dom_total
        gap = abs(dom_acc - 0.5)

        if scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch + 1}] Total Loss: {avg_total_loss:.4f} | "
              f"Cls: {avg_cls_loss:.4f} | Dom: {avg_dom_loss:.4f} | "
              f"DomAcc: {dom_acc:.4f}")


        if gap < 0.05 and avg_cls_loss < 0.05 and epoch > 10:
            patience +=1
            if gap < best_gap:
                best_gap = gap
                best_model_state = copy.deepcopy(model.state_dict())
            print(f"[INFO] patience {patience} / 3")
            if patience > 3:
                model.load_state_dict(best_model_state)
                print("[INFO] Early stopping: domain aligned and classifier converged.")
                break
        else:
            patience = 0
            best_gap = gap

        if best_model_state is not None:
            # torch.save(best_model_state, os.path.join(out_path, 'test_best_model.pth'))
            model.load_state_dict(best_model_state)


    return model






if __name__ == '__main__':
    set_seed(seed=42)
    with open("../configs/default.yaml", 'r') as f:
        config = yaml.safe_load(f)['baseline']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_layers = config['num_layers']
    kernel_size = config['kernel_size']
    start_channels = config['start_channels']
    num_epochs = config['num_epochs']

    source_path = '../datasets/source/train/DC_T197_RP.txt'
    target_path = '../datasets/HC_T185_RP.txt'
    target_test_path = '../datasets/HC_T185_RP.txt'
    out_path = 'model'
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Flexible_DANN(num_layers=num_layers,
                          start_channels=start_channels,
                          kernel_size=kernel_size,
                          cnn_act='leakrelu',
                          num_classes=10,
                          lambda_=0.5).to(device)

    source_loader, target_loader = get_dataloaders(source_path, target_path, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.1
    )
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    print("[INFO] Starting standard DANN training (no pseudo labels)...")
    model=train_dann(model, source_loader, target_loader,
               optimizer, criterion_cls, criterion_domain,
               device, num_epochs=20, lambda_=0.5,scheduler=scheduler)

    print("[INFO] Evaluating on target test set...")
    test_dataset = PKLDataset(target_test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    general_test_model(model, criterion_cls, test_loader, device)
