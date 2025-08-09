import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import random
import yaml
from models.Flexible_DANN import Flexible_DANN
from PKLDataset import PKLDataset
from models.pseudo_train_and_test import pseudo_test_model
from utils.get_no_label_dataloader import get_target_loader

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

def train_dann(model, source_loader, target_loader,
               optimizer, criterion_cls, criterion_domain,
               device, num_epochs=20, lambda_=0.1):
    model.train()
    for epoch in range(num_epochs):
        total_loss, total_cls_loss, total_dom_loss = 0.0, 0.0, 0.0
        dom_correct, dom_total = 0, 0
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

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_dom_loss += loss_dom.item()

        dom_acc = dom_correct / dom_total
        avg_cls_loss = total_cls_loss / len(source_loader)

        print(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f} | "
              f"Cls: {avg_cls_loss:.4f} | Dom: {total_dom_loss:.4f} | "
              f"DomAcc: {dom_acc:.4f}")

        if 0.48 < dom_acc < 0.52 and avg_cls_loss < 0.5 and epoch > 10:
            print("[INFO] Early stopping: domain aligned and classifier converged.")
            break

    return model






if __name__ == '__main__':
    # set_seed(seed=42)
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
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    print("[INFO] Starting standard DANN training (no pseudo labels)...")
    train_dann(model, source_loader, target_loader,
               optimizer, criterion_cls, criterion_domain,
               device, num_epochs=30, lambda_=0.5)

    print("[INFO] Evaluating on target test set...")
    test_dataset = PKLDataset(target_test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pseudo_test_model(model, criterion_cls, test_loader, device)
