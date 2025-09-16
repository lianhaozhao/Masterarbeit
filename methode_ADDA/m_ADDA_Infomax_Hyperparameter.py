import os, json, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import optuna
import math
from models.Flexible_ADDA import Flexible_ADDA,freeze,unfreeze,DomainClassifier
from PKLDataset import PKLDataset
from utils.general_train_and_test import general_test_model
from models.get_no_label_dataloader import get_dataloaders,get_pseudo_dataloaders
import yaml
from models.generate_pseudo_labels_with_LMMD import generate_pseudo_with_stats
from models.MMD import classwise_mmd_biased_weighted,suggest_mmd_gammas
from collections import deque


def adam_param_groups(named_params, weight_decay):
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):  # BN/LayerNorm 权重 & bias 不做 decay
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

@torch.no_grad()
def copy_encoder_params(src_model, tgt_model, device):
    tgt_model.load_state_dict(copy.deepcopy(src_model.state_dict()))
    tgt_model.to(device)
# ------------------ InfoMax (target domain) ------------------
@torch.no_grad()
def _safe_mean_prob(p, eps=1e-8):
    p = p.clamp_min(eps)
    return p / p.sum(dim=1, keepdim=True)

def entropy_mean(p, eps=1e-8):
    # E_x[ H(p_x) ]
    p = p.clamp_min(eps)
    return (-p * p.log()).sum(dim=1).mean()

def entropy_marginal(p, eps=1e-8):
    # H( E_x[p_x] )
    p_bar = p.mean(dim=0)
    p_bar = p_bar.clamp_min(eps)
    return -(p_bar * p_bar.log()).sum()

def infomax_loss_from_logits(logits, T=1.0, marg_weight=1.0):
    """
        依据 “信息最大化” (InfoMax) 思想，从分类 logits 计算无监督正则项：
        I(z; ŷ) = H(ŷ) - H(ŷ|z)。
        训练中最小化的目标为：
            L = H(ŷ|z) - w * H(ŷ)
        其中 H(ŷ|z) 是条件熵（鼓励单样本预测更自信），H(ŷ) 是边际熵（鼓励整体类别使用均衡，防止塌缩）。

        参数
        ----
        logits : Tensor
            分类头输出的未归一化得分，形状 [B, C]。
        T : float, 默认 1.0
            Softmax 温度。T > 1 使分布变“软”（置信度下降），T < 1 使分布变“尖”（置信度上升）。
            会同时影响条件熵与边际熵的数值。
        marg_weight : float, 默认 1.0
            边际熵权重 w。数值越大，越强烈地抑制“塌缩到单一类别”的解。
            常见范围 0.5 ~ 2.0，可据验证集曲线调参。

        返回
        ----
        loss : Tensor (标量，requires_grad=True)
            最小化目标：H(ŷ|z) - w * H(ŷ)。用于反向传播。
        h_cond_detached : Tensor (标量，no grad)
            条件熵 H(ŷ|z) 的经验估计（E_x[-∑_c p(c|x) log p(c|x)]）。仅用于日志监控。
        h_marg_detached : Tensor (标量，no grad)
            边际熵 H(ŷ) 的经验估计，其中 ŷ 的边际分布为 p̄ = E_x[p(·|x)]。
            数学形式：H(ŷ) = -∑_c p̄_c log p̄_c。仅用于日志监控。

        """
    # I(z;ŷ) = H(ŷ) - H(ŷ|z); minimiere  H(ŷ|z) - w * H(ŷ)
    p = F.softmax(logits / T, dim=1)  # [B, C], Klassenwahrscheinlichkeiten pro Beispiel
    h_cond = entropy_mean(p)  # Skalar, Schätzung der bedingten Entropie
    h_marg = entropy_marginal(p)  # Schätzung der marginalen Entropie
    return h_cond - marg_weight * h_marg, h_cond.detach(), h_marg.detach()


# ===== 新增：阶段1 —— 仅用源域训练 (F_s + C) =====
def pretrain_source_classifier(
        src_model,
        source_loader,
        optimizer,lr_d,weight_decay,
        criterion_cls,
        device,
        num_epochs=5,
        scheduler=None,
):
    """
    - 源域：交叉熵 -> 更新 feature_extractor + classifier
    - 目标域：InfoMax 仅更新 classifier（对 feature_extractor 用 eval()+no_grad 提特征）
    - 可选：epoch 末再做一遍 target-only InfoMax，扩大目标覆盖
    """
    src_model.train()

    for epoch in range(num_epochs):
        tot_loss = tot_n = 0.0

        # 源域 CE + 目标域 InfoMax（只训分类头）的混合小循环
        for xb, yb in source_loader:
            xb, yb = xb.to(device), yb.to(device)

            # 1) 源域监督 CE：正常 forward（允许更新提取器与分类头）
            logits_s, _ = src_model(xb)
            loss = criterion_cls(logits_s, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            tot_loss += loss.item() * bs
            tot_n += bs


        if scheduler is not None:
            scheduler.step()

    return src_model

# ===== 新增：阶段2 —— ADDA 对抗 + InfoMax  =====
def train_adda_infomax_lmmd(
    src_model, tgt_model, source_loader, target_loader,
    device, num_epochs=20, num_classes=10,batch_size=16,
    # 判别器/优化器
    lr_ft=1e-4, lr_d=1e-4, wd=0.0,d_steps =1 ,ft_steps =1 ,
    # InfoMax
    im_T=1.0, im_weight=0.5, im_marg_w=1.0,
):
    # 1) 冻结源模型 (完全固定 F_s + C)
    src_model.eval(); freeze(src_model)

    # 2) 冻结目标模型的 classifier，只训练其 encoder
    for p in tgt_model.classifier.parameters():
        p.requires_grad = False
    enc_named_params = []
    for n, p in tgt_model.named_parameters():
        if ("feature_extractor" in n) or ("feature_reducer" in n):
            enc_named_params.append((n, p))

    opt_ft = torch.optim.Adam(
        adam_param_groups(enc_named_params, wd),
        lr=lr_ft
    )

    # 3) 通过一个 batch 推断特征维度，按需构造 D
    with torch.no_grad():
        xb_s, yb_s = next(iter(source_loader))
        xb_s = xb_s.to(device)
        _ , feat_s= src_model(xb_s)
        feat_dim = feat_s.size(1)
    D = DomainClassifier(feature_dim=feat_dim).to(device)
    opt_d = torch.optim.Adam(
        adam_param_groups(D.named_parameters(), wd),
        lr=lr_d
    )
    c_dom = nn.CrossEntropyLoss().to(device)

    best_loss = float("inf")
    best_state = None

    # 4) 训练循环（交替优化 D 和 F_t）
    for epoch in range(num_epochs):

        it_src, it_tgt = iter(source_loader), iter(target_loader)
        len_src, len_tgt = len(source_loader), len(target_loader)
        steps = max(len_src, len_tgt)
        #新的迭代器
        it_tgt_ft = iter(target_loader)

        tgt_model.train()
        tgt_model.classifier.eval()
        D.train()

        # 统计
        d_loss_sum = g_loss_sum = im_loss_sum = ft_loss_sum = 0.0
        d_acc_sum = 0.0; d_cnt = 0

        for _ in range(steps):
            try: xs, ys = next(it_src)
            except StopIteration:
                it_src = iter(source_loader); xs, ys = next(it_src)
            try: xt = next(it_tgt)
            except StopIteration:
                it_tgt = iter(target_loader); xt = next(it_tgt)
            if isinstance(xt, (tuple,list)): xt = xt[0]
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            #(A) 训练 D: max log D(F_s(xs)) + log (1 - D(F_t(xt)))
            for _k in range(d_steps):
                with torch.no_grad():
                    _, f_s= src_model(xs)       # [B, d]
                    _, f_t= tgt_model(xt)       # [B, d]
                d_in  = torch.cat([f_s, f_t], dim=0)
                d_lab = torch.cat([torch.ones(f_s.size(0)), torch.zeros(f_t.size(0))], dim=0).long().to(device)  # 1=source,0=target
                d_out = D(d_in)
                loss_d = c_dom(d_out, d_lab)
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                xt_last = xt.detach()

                # 记录 D acc
                with torch.no_grad():
                    pred = d_out.argmax(1)
                    d_acc = (pred == d_lab).float().mean().item()
                    d_acc_sum += d_acc; d_cnt += 1
                    d_loss_sum += loss_d.item()

            # (B) 训练 F_t: min 交叉熵(D(F_t(xt)), “source”标签)
            D.eval()
            for p in D.parameters():
                p.requires_grad = False


            for _k in range(ft_steps):
                if _k == 0 and xt_last is not None:
                    xt_ft = xt_last.to(device)  # 复用 D 的最后一个 batch
                else:
                    try:
                        xt_ft = next(it_tgt_ft)
                    except StopIteration:
                        it_tgt_ft = iter(target_loader)
                        xt_ft = next(it_tgt_ft)
                xt_ft = xt_ft.to(device)
                _, f_t  = tgt_model(xt_ft)
                fool_lab = torch.ones(f_t.size(0), dtype=torch.long, device=device)  # 让 D 预测成“source” -1
                g_out = D(f_t)
                loss_g = c_dom(g_out, fool_lab)

                # InfoMax 正则 —— 更自信但不塌缩
                logits_t = tgt_model.classifier(f_t)
                loss_im, h_cond, h_marg = infomax_loss_from_logits(logits_t, T=im_T, marg_weight=im_marg_w)
                loss_im = im_weight * loss_im

                loss_ft = loss_g + loss_im
                opt_ft.zero_grad()
                loss_ft.backward()
                opt_ft.step()

                def to_scalar(x):
                    return x.detach().item() if torch.is_tensor(x) else float(x)

                g_loss_sum += to_scalar(loss_g)
                im_loss_sum += to_scalar(loss_im)
                ft_loss_sum += to_scalar(loss_ft)
            for p in D.parameters():
                p.requires_grad = True
            D.train()
        if epoch > num_epochs//2:
            if im_loss_sum < best_loss:
                best_loss = im_loss_sum
                best_state = copy.deepcopy(tgt_model.state_dict())


        if best_state is not None:
            tgt_model.load_state_dict(best_state)

    return tgt_model,D


# 只搜对抗专属超参
def suggest_adv_only(trial):
    params = {
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64]),  # 可改成 [32, 48, 64]
        "learning_rate_pre": trial.suggest_float("learning_rate_pre", 1e-4, 1e-3, log=True),
        "weight_decay_pre": trial.suggest_float("weight_decay_pre", 1e-6, 1e-4, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "im_weight":     trial.suggest_categorical("im_weight", [0.8]),
        "im_T":          trial.suggest_categorical("im_T", [1.0])
    }
    return params


@torch.no_grad()
def infomax_unsup_score_from_loader(model, D,target_loader, device,
                                    T=1.0, marg_weight=1.0, eps=1e-8):
    model.eval()
    probs = []
    for xb in target_loader:
        xb = xb.to(device)
        cls_out, _ = model(xb)
        p = F.softmax(cls_out / T, dim=1)
        probs.append(p.detach().cpu())
    if not probs:
        return {"score": float("inf"), "domain_gap": 0.5}
    P = torch.cat(probs, dim=0).clamp_min(eps)
    mean_entropy = (-P * P.log()).sum(dim=1).mean().item()
    p_bar = P.mean(dim=0).clamp_min(eps)
    marginal_entropy = (-(p_bar * p_bar.log()).sum()).item()
    score = mean_entropy - marg_weight * marginal_entropy

    # 域不可分性（gap 越小越好）
    tot, correct = 0, 0
    D.eval()
    for xt in target_loader:
        xt = xt.to(device)
        _, f_t= model(xt)
        g_out = D(f_t)
        g_out = g_out.argmax(dim=1)
        fool_lab = torch.ones(f_t.size(0), dtype=torch.long, device=device)
        correct += (g_out == fool_lab).sum().item()
        tot += xt.size(0)
    domain_acc = (correct / max(1, tot)) if tot > 0 else 1.0
    domain_gap = abs(domain_acc - 0.5)
    return {"score": float(score), "domain_gap": float(domain_gap)}



def objective_adv_only(trial,
                       source_train=None,
                       target_train=None,
                       out_dir     =None,
                       ):
    with open("../configs/default.yaml", 'r') as f:
        cfg = yaml.safe_load(f)['DANN_LMMD_INFO']
    num_layers = cfg['num_layers']; ksz = cfg['kernel_size']; sc = cfg['start_channels']
    num_epochs = 20
    source_epoch=6
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = suggest_adv_only(trial)

    # 数据（batch 固定）
    src_loader, tgt_loader = get_dataloaders(source_train, target_train, batch_size=p["batch_size"])

    # 模型（结构固定）
    src_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                              cnn_act='leakrelu', num_classes=10).to(device)
    optimizer_src = torch.optim.Adam(
        adam_param_groups(src_model.named_parameters(), p["weight_decay_pre"]),
        lr=p["learning_rate_pre"], betas=(0.9, 0.999), eps=1e-8
    )
    scheduler_src = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_src, T_max=5, eta_min=p["learning_rate_pre"] * 0.1)
    src_cls = nn.CrossEntropyLoss()

    src_model = pretrain_source_classifier(src_model, src_loader, optimizer_src, p["learning_rate_pre"], p["weight_decay_pre"], src_cls, device,
                               num_epochs=source_epoch, scheduler=scheduler_src)

    tgt_model = Flexible_ADDA(num_layers=num_layers, start_channels=sc, kernel_size=ksz,
                              cnn_act='leakrelu', num_classes=10).to(device)
    copy_encoder_params(src_model, tgt_model, device)

    tgt_model,D = train_adda_infomax_lmmd(
        src_model, tgt_model, src_loader, tgt_loader, device,
        num_epochs=num_epochs, num_classes=10, batch_size=p['batch_size'],
        # 判别器/优化器
        lr_ft=p["learning_rate"], lr_d=p["learning_rate"]*2, wd=p["weight_decay"], d_steps=1, ft_steps=2,
        # InfoMax
        im_T=1.0, im_weight=0.8, im_marg_w=1.0,
    )

    metrics = infomax_unsup_score_from_loader(tgt_model, D, tgt_loader, device,
                                              T=1.0, marg_weight=1.0)

    trial.set_user_attr("score", metrics["score"])
    trial.set_user_attr("domain_gap", metrics["domain_gap"])

    print(f"[InfoMax] score={metrics['score']:.4f} | "
          f"gap={metrics['domain_gap']:.3f}")

    # 记录 trial
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

    # 2目标：最小化 InfoMax 分数 最小化 domain_gap
    return float(metrics["score"]), float(metrics["domain_gap"])


# ========= 主程序 =========
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 直接搜索对抗专属超参
    sampler = optuna.samplers.NSGAIISampler()
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    study.optimize(lambda t: objective_adv_only(
        t,
        source_train='../datasets/source/train/DC_T197_RP.txt',
        target_train='../datasets/target/train/HC_T185_RP.txt',
        out_dir     ='../datasets/info_optuna_ADDA',
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
    out_dir = '../datasets/info_optuna_ADDA'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "best_adv_only.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("[ADDA-AdvOnly] Best:", best)
