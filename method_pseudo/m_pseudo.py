import torch
import torch.nn as nn
from models.Flexible_CNN import Flexible_CNN,freeze_feature_train_head
from models.generate_pseudo_labels import generate_pseudo_labels,generate_soft_pseudo_labels
from PKLDataset import PKLDataset
from torch.utils.data import DataLoader ,TensorDataset
import os
import numpy as np
import random
import yaml
from models.pseudo_train_and_test import pseudo_train_model, pseudo_test_model, pseudo_soft_train_model

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

with open("../configs/default.yaml", 'r') as f:
    config = yaml.safe_load(f)['baseline']
# 提取参数
batch_size = config['batch_size']
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
num_layers = config['num_layers']
kernel_size = config['kernel_size']
start_channels = config['start_channels']
num_epochs = config['num_epochs']
early_stopping_patience = config['early_stopping_patience']

class NoLabelDataset(torch.utils.data.Dataset):
    """
       A wrapper dataset that removes labels from a labeled dataset.

       Args:
           signal_dataset (Dataset): A dataset returning (signal, label) tuples.
       """
    def __init__(self, signal_dataset):
        self.signal_dataset = signal_dataset

    def __len__(self):
        return len(self.signal_dataset)

    def __getitem__(self, idx):
        signal, _ = self.signal_dataset[idx]
        return signal


def get_target_loader(path = None ,batch_size=64):
    """
        Loads unlabeled target domain data as a DataLoader for inference or pseudo-labeling.

        This function reads the target dataset from the given txt path,
        removes labels using NoLabelDataset, and returns a DataLoader that yields only input signals.

        Args:
            path (str): Path to the txt file listing .pkl sample paths (target domain).
            batch_size (int): Batch size for the DataLoader.

        Returns:
            DataLoader: A DataLoader yielding batches of input signals without labels.
        """

    dataset = PKLDataset(txt_path=path)
    no_label_dataset = NoLabelDataset(dataset)
    loader = DataLoader(no_label_dataset, batch_size=batch_size, shuffle=False)
    return loader

if __name__ == '__main__':
    # initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Flexible_CNN(num_layers=num_layers,
                                 start_channels=start_channels,
                                 kernel_size=kernel_size,
                                 cnn_act='leakrelu',
                                 num_classes=10).to(device)


    # out file path
    out_path = "model"
    os.makedirs(out_path, exist_ok=True)
    # The address of the model
    model_path = '../baseline/model/best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("[INFO] Model loaded")

    # Get unlabeled data of the target domain
    target_loader = get_target_loader(path= '../datasets/target/train/HC_T185_RP.txt',batch_size=batch_size)

    # Using a pseudo-soft label generator
    # pseudo_data, pseudo_labels = generate_pseudo_labels(model, target_loader, device, threshold=0.9)

    # pseudo_data, pseudo_probs = generate_soft_pseudo_labels(model, target_loader, device)
    # pseudo_dataset = TensorDataset(pseudo_data, pseudo_probs)
    # pseudo_loader = DataLoader(pseudo_dataset, batch_size=64, shuffle=True)

    # Using a pseudo-hard label generator
    pseudo_data, pseudo_labels = generate_pseudo_labels(model, target_loader, device, threshold=0.95)
    # Wrap as a Dataset
    pseudo_dataset = TensorDataset(pseudo_data, pseudo_labels)
    # Convert to DataLoader (can be used for training)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    # # Freeze feature extraction layer parameters
    #
    # optimizer = freeze_feature_train_head(model, learning_rate, weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3
    )
    model,best_val_loss= pseudo_train_model(
            model, pseudo_loader, optimizer,criterion, device,
            num_epochs=num_epochs, early_stopping_patience=early_stopping_patience,scheduler=scheduler,out_path=out_path
        )
    pseudo_test_dataset = PKLDataset('../datasets/target/test/HC_T185_RP.txt')
    pseudo_test_loader = DataLoader(pseudo_test_dataset, batch_size=batch_size, shuffle=False)

    pseudo_test_model(model, criterion, pseudo_test_loader, device)