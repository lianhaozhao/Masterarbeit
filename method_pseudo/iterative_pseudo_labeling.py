import torch
import torch.nn as nn
from models.Flexible_CNN import Flexible_CNN, freeze_feature_train_head
from models.generate_pseudo_labels import generate_pseudo_labels
from PKLDataset import PKLDataset
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import random
import yaml
from models.pseudo_train_and_test import pseudo_train_model, pseudo_test_model

# --------------------------- Seed ---------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# --------------------------- Config ---------------------------
with open("../configs/default.yaml", 'r') as f:
    config = yaml.safe_load(f)['baseline']

batch_size = config['batch_size']
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
num_layers = config['num_layers']
kernel_size = config['kernel_size']
start_channels = config['start_channels']
num_epochs = config['num_epochs']
early_stopping_patience = config['early_stopping_patience']

# --------------------------- Dataset ---------------------------
class NoLabelDataset(torch.utils.data.Dataset):
    """A wrapper dataset that removes labels from a labeled dataset."""
    def __init__(self, signal_dataset):
        self.signal_dataset = signal_dataset

    def __len__(self):
        return len(self.signal_dataset)

    def __getitem__(self, idx):
        signal, _ = self.signal_dataset[idx]
        return signal

def get_target_loader(path=None, batch_size=batch_size):
    dataset = PKLDataset(txt_path=path)
    no_label_dataset = NoLabelDataset(dataset)
    return DataLoader(no_label_dataset, batch_size=batch_size, shuffle=False)

# --------------------------- Iterative Pseudo Label Training ---------------------------
def iterative_pseudo_label_training(
    model,
    target_txt_path,
    generate_fn,
    train_fn,
    optimizer,
    criterion,
    device,
    scheduler=None,
    num_rounds=3,
    threshold=0.9,
    batch_size=64,
    out_path="model"
):
    os.makedirs(out_path, exist_ok=True)

    for round_idx in range(num_rounds):
        print(f"\n Pseudo-Labeling Round {round_idx+1}/{num_rounds}")

        # get unlabeled target loader
        target_loader = get_target_loader(path=target_txt_path, batch_size=batch_size)

        # generate pseudo-labels
        pseudo_data, pseudo_labels = generate_fn(model, target_loader, device, threshold=threshold)
        if pseudo_data.shape[0] == 0:
            print("No pseudo-labels with sufficient confidence. Skipping.")
            break

        pseudo_dataset = TensorDataset(pseudo_data, pseudo_labels)
        pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)

        # Train on pseudo-labeled data
        model, best_loss = train_fn(
            model=model,
            pseudo_loader=pseudo_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            scheduler=scheduler,
            out_path=out_path
        )
        test_dataset = PKLDataset('../datasets/target/test/HC_T185_RP.txt')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        pseudo_test_model(model, criterion, test_loader, device)

    return model

# --------------------------- Main ---------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Flexible_CNN(
        num_layers=num_layers,
        start_channels=start_channels,
        kernel_size=kernel_size,
        cnn_act='leakrelu',
        num_classes=10
    ).to(device)

    # Load source model
    model_path = '../baseline/model/best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("[INFO] Loaded pretrained model.")

    # # Freeze backbone, train head only
    # optimizer = freeze_feature_train_head(model, learning_rate, weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    criterion = nn.CrossEntropyLoss()

    # Iterative pseudo-labeling training
    model = iterative_pseudo_label_training(
        model=model,
        target_txt_path='../datasets/target/train/HC_T185_RP.txt',
        generate_fn=generate_pseudo_labels,
        train_fn=pseudo_train_model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        num_rounds=3,
        threshold=0.95,
        batch_size=batch_size,
        out_path="model"
    )

    # Evaluate on test set
    test_dataset = PKLDataset('../datasets/target/test/HC_T185_RP.txt')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pseudo_test_model(model, criterion, test_loader, device)
