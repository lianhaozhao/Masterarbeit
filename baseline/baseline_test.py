import os
import torch
import torch.nn as nn
import numpy as np
from models.Flexible_CNN import Flexible_CNN
from PKLDataset import PKLDataset
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm


# Load configuration
with open("../configs/default.yaml", 'r') as f:
    config = yaml.safe_load(f)['baseline_2']

batch_size = config['batch_size']
num_layers = config['num_layers']
kernel_size = config['kernel_size']
start_channels = config['start_channels']

# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i in range(10):
    model_path = f'model/ts/run_{i}/best_model.pth'
    print(f"run{i}start")
    # file = ['../datasets/HC_T197_RP.txt','../datasets/HC_T194_RP.txt','../datasets/HC_T191_RP.txt','../datasets/HC_T188_RP.txt','../datasets/HC_T185_RP.txt']
    file = ['../datasets/DC_T194_RP.txt', '../datasets/DC_T191_RP.txt',
            '../datasets/DC_T188_RP.txt', '../datasets/DC_T185_RP.txt']

    for item in file:
        # load data
        test_dataset = PKLDataset(item)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model
        model = Flexible_CNN(
            num_layers=num_layers,
            start_channels=start_channels,
            kernel_size=kernel_size,
            cnn_act='leakrelu',
            num_classes=10
        ).to(device)

        # Loading model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Evaluation Model
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_loss /= total
        test_accuracy = correct / total

        print(f"\nfile:{item} \nTest Loss: {test_loss:.6f} Test Accuracy: {test_accuracy:.4f}")

