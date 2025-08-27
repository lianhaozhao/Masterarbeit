import copy, math, random, os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Flexible_ADDA import Flexible_ADDA, freeze, unfreeze, DomainClassifier
from PKLDataset import PKLDataset
from models.get_no_label_dataloader import get_target_loader
from utils.general_train_and_test import general_test_model

# ========== 基础工具 ==========
def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_dataloaders(source_path, target_path, batch_size):
    src_ds = PKLDataset(txt_path=source_path)
    tgt_loader = get_target_loader(target_path, batch_size=batch_size, shuffle=True)
    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True)
    return src_loader, tgt_loader

# ========== 阶段1：源域监督预训练 (Fs + C) ==========
def pretrain_source_classifier(src_model, source_loader, optimizer, criterion_cls, device, num_epochs=5, scheduler=None):
    src_model.train()
    for epoch in range(num_epochs):
        tot_loss, tot_n = 0.0, 0
        for xb, yb in source_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _, _ = src_model(xb)
            loss = criterion_cls(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * xb.size(0); tot_n += xb.size(0)
        print(f"[SRC PRETRAIN] Epoch {epoch+1}/{num_epochs} | cls:{tot_loss/max(1,tot_n):.4f}")
        if scheduler is not None: scheduler.step()
    return src_model

# ========== 阶段2： ADDA ==========
def train_adda_pure(
    src_model, tgt_model, source_loader, target_loader,
    device, num_epochs=20, lr_ft=1e-3, lr_d=5e-4, weight_decay=0.0,
    d_steps=1, ft_steps=1
):
    # 1) 冻结源模型（Fs + C）并固定到 eval
    src_model.eval(); freeze(src_model)

    # 2) 冻结目标模型的 classifier（只训练 encoder），并把 classifier 设为 eval 防止 BN/Dropout 波动
    for p in tgt_model.classifier.parameters():
        p.requires_grad = False
    tgt_model.classifier.eval()

    # 只拿 encoder 参数给优化器
    enc_params = [p for n, p in tgt_model.named_parameters() if not n.startswith('classifier')]
    opt_ft = torch.optim.Adam(enc_params, lr=lr_ft, weight_decay=weight_decay)

    # 3) 构造域判别器 D（输入是特征）
    with torch.no_grad():
        xb_s, _ = next(iter(source_loader))
        xb_s = xb_s.to(device)
        _, feat_s, _ = src_model(xb_s)
        feat_dim = feat_s.size(1)
    D = DomainClassifier(feature_dim=feat_dim).to(device)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, weight_decay=weight_decay)
    c_dom = nn.CrossEntropyLoss().to(device)

    # 4) 训练循环（交替优化 D 和 Ft）
    for epoch in range(num_epochs):
        tgt_model.train()
        D.train()

        it_src, it_tgt = iter(source_loader), iter(target_loader)
        steps = max(len(source_loader), len(target_loader))

        d_loss_sum = g_loss_sum = 0.0
        d_acc_sum, d_cnt = 0.0, 0

        for _ in range(steps):
            # ------- 取 batch -------
            try: xs, ys = next(it_src)
            except StopIteration:
                it_src = iter(source_loader); xs, ys = next(it_src)
            try: xt = next(it_tgt)
            except StopIteration:
                it_tgt = iter(target_loader); xt = next(it_tgt)
            if isinstance(xt, (tuple, list)): xt = xt[0]
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            # ------- (A) 训练 D：分辨源/目标特征 -------
            for _k in range(d_steps):
                with torch.no_grad():
                    _, f_s, _ = src_model(xs)   # 源特征（冻结）
                    _, f_t, _ = tgt_model(xt)   # 目标特征（当前 Ft）
                d_in  = torch.cat([f_s, f_t], dim=0)
                d_lab = torch.cat([
                    torch.ones(f_s.size(0), dtype=torch.long, device=device),   # 1=source
                    torch.zeros(f_t.size(0), dtype=torch.long, device=device)   # 0=target
                ], dim=0)
                d_out  = D(d_in)
                loss_d = c_dom(d_out, d_lab)
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()

                with torch.no_grad():
                    pred = d_out.argmax(1)
                    d_acc = (pred == d_lab).float().mean().item()
                    d_acc_sum += d_acc; d_cnt += 1
                    d_loss_sum += loss_d.item()

            # ------- (B) 训练 Ft：愚弄 D（让 D 把目标当“源”） -------
            D.eval()
            for p in D.parameters(): p.requires_grad = False

            for _k in range(ft_steps):
                _, f_t, _ = tgt_model(xt)                  # 仅 encoder 会更新
                g_out  = D(f_t)
                fool_lab = torch.ones(f_t.size(0), dtype=torch.long, device=device)  # 目标→源
                loss_g = c_dom(g_out, fool_lab)
                opt_ft.zero_grad(); loss_g.backward(); opt_ft.step()
                g_loss_sum += loss_g.item()

            for p in D.parameters(): p.requires_grad = True
            D.train()

        print(f"[ADDA-PURE] Epoch {epoch+1}/{num_epochs} | "
              f"D_loss:{d_loss_sum/max(1,steps*d_steps):.4f} | "
              f"G_loss:{g_loss_sum/max(1,steps*ft_steps):.4f} | "
              f"D-acc:{(d_acc_sum/max(1,d_cnt)):.4f}")

        # ------- 每个 epoch 结束做一次目标测试评估 -------
        print("[INFO] Evaluating on target test set...")
        target_test_path = '../datasets/target/test/HC_T197_RP.txt'
        test_dataset = PKLDataset(target_test_path)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        cls_criterion = nn.CrossEntropyLoss()
        general_test_model(tgt_model, cls_criterion, test_loader, device)

    return tgt_model

# ========== 主程序 ==========
if __name__ == "__main__":
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    bs = cfg['batch_size']; lr = cfg['learning_rate']; wd = cfg['weight_decay']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = cfg['num_epochs']

    src_path = '../datasets/source/train/DC_T197_RP.txt'
    tgt_path = '../datasets/target/train/HC_T191_RP.txt'
    tgt_test = '../datasets/target/test/HC_T191_RP.txt'

    for run_id in range(5):
        print(f"\n========== RUN {run_id} (ADDA-PURE) ==========")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建源模型 & 数据
        src_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                  cnn_act='leakrelu', num_classes=10, lambda_=1.0).to(device)
        src_loader, tgt_loader = get_dataloaders(src_path, tgt_path, bs)
        optimizer_src = torch.optim.Adam(src_model.parameters(), lr=lr, weight_decay=wd)
        scheduler_src = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_src, T_max=num_epochs, eta_min=lr*0.1)
        src_cls = nn.CrossEntropyLoss()

        print("[INFO] SRC pretrain (Fs + C) ...")
        pretrain_source_classifier(src_model, src_loader, optimizer_src, src_cls, device,
                                   num_epochs=max(1, num_epochs//4), scheduler=scheduler_src)

        # 目标模型从源模型拷贝初始化
        tgt_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                                  cnn_act='leakrelu', num_classes=10, lambda_=0.0).to(device)
        tgt_model.load_state_dict(copy.deepcopy(src_model.state_dict()))
        tgt_model.to(device)

        print("[INFO] ADDA stage (Ft vs D, PURE) ...")
        tgt_model = train_adda_pure(
            src_model, tgt_model, src_loader, tgt_loader, device,
            num_epochs=num_epochs,
            lr_ft=lr, lr_d=lr*0.5, weight_decay=wd,
            d_steps=1, ft_steps=3
        )

        # 最终评测（目标测试集）
        print("[INFO] Evaluating on target test set...")
        test_ds = PKLDataset(tgt_test)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
        cls_criterion = nn.CrossEntropyLoss().to(device)
        general_test_model(tgt_model, cls_criterion, test_loader, device)

        # 清理
        del src_model, tgt_model, optimizer_src, scheduler_src, src_loader, tgt_loader, test_loader, test_ds
        if torch.cuda.is_available(): torch.cuda.empty_cache()
