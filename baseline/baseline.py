import os
import torch
import torch.nn as nn
import numpy as np
from models.Flexible_CNN import Flexible_CNN
from PKLDataset import PKLDataset
from torch.utils.data import DataLoader
import yaml
import random
from tqdm import tqdm
from n_PKLDataset import N_PKLDataset,fit_normalizer_from_txt
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

def adam_param_groups(model, wd):
    """
    Build parameter groups for Adam/AdamW with and without weight decay.

    This function splits model parameters into two groups:
      1. Parameters that use weight decay (e.g., weights of linear/conv layers)
      2. Parameters that do NOT use weight decay (e.g., normalization weights and biases)

    Rules:
      - Skip parameters that do not require gradients (p.requires_grad == False)
      - 1D parameters (p.ndim == 1), typically normalization layer weights (LayerNorm/GroupNorm/BatchNorm),
        are put into the no_decay group
      - Parameters whose names end with ".bias" are put into the no_decay group
      - All other parameters are put into the decay group

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        wd (float): Weight decay factor applied to parameters in the decay group.

    Returns:
        list[dict]: A list of parameter groups suitable for torch.optim.Adam/AdamW, e.g.:
            [
                {"params": decay_params, "weight_decay": wd},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith('.bias'):  # norm weights & biases: no weight decay
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]




with open("../configs/default.yaml", 'r') as f:
    config = yaml.safe_load(f)['DANN_LMMD_INFO']
# Extract parameters
batch_size = config['batch_size']
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
num_layers = config['num_layers']
kernel_size = config['kernel_size']
start_channels = config['start_channels']
num_epochs = config['num_epochs']
early_stopping_patience = config['early_stopping_patience']

def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                num_epochs=20, early_stopping_patience=3, scheduler=None, out_path=None):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = inputs.size(0)
            train_loss += loss.item() * batch_size_actual
            total_train_samples += batch_size_actual

            # Accuracy calculation (training)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()

        train_loss /= total_train_samples
        train_accuracy = correct_train / total_train_samples

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(outputs, labels)

                batch_size_actual = inputs.size(0)
                val_loss += loss.item() * batch_size_actual
                total_val_samples += batch_size_actual

                # Accuracy calculation (validation)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()

        val_loss /= total_val_samples
        val_accuracy = correct_val / total_val_samples

        if scheduler is not None:
            scheduler.step(val_loss)

        if (epoch+1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"- Train Loss: {train_loss:.6f}, Acc: {train_accuracy:.4f} "
                  f"- Val Loss: {val_loss:.6f}, Acc: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(f"Patience Counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience and epoch > num_epochs * 0.25:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(out_path, 'best_model.pth'))
        model.load_state_dict(best_model_state)

    return best_val_loss

if __name__ == '__main__':

    for i in range(10):
        train_dataset = PKLDataset('../datasets/source/train/DC_T197_RP.txt')
        val_dataset = PKLDataset('../datasets/source/validation/DC_T197_RP.txt')
        out_path = f"model/ts/run_{i}"
        os.makedirs(out_path, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = Flexible_CNN(num_layers=num_layers,
                                 start_channels=start_channels,
                                 kernel_size=kernel_size,
                                 cnn_act='leakrelu',
                                 num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            adam_param_groups(model, weight_decay),
            lr=learning_rate, betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3
        )
        best_val_loss= train_model(
                model, train_loader, val_loader, optimizer, criterion, device,
                num_epochs=num_epochs, early_stopping_patience=early_stopping_patience,scheduler=scheduler,out_path=out_path
            )



