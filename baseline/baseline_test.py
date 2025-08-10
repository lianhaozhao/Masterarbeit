import os
import torch
import torch.nn as nn
import numpy as np
from models.Flexible_CNN import Flexible_CNN
from PKLDataset import PKLDataset
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm


# 加载配置
with open("../configs/default.yaml", 'r') as f:
    config = yaml.safe_load(f)['baseline']

batch_size = config['batch_size']
num_layers = config['num_layers']
kernel_size = config['kernel_size']
start_channels = config['start_channels']

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file = ['../datasets/DC_T185_RP.txt']
for item in file:
    # 加载测试数据
    test_dataset = PKLDataset(item)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = Flexible_CNN(
        num_layers=num_layers,
        start_channels=start_channels,
        kernel_size=kernel_size,
        cnn_act='leakrelu',
        num_classes=10
    ).to(device)

    # 加载模型权重
    model_path = 'model/test_best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 评估模型
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss /= total
    test_accuracy = correct / total

    print(f"\nfile:{item} \nTest Loss: {test_loss:.6f} Test Accuracy: {test_accuracy:.4f}")

