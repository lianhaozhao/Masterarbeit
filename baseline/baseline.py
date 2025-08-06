import os
import torch
import torch.nn as nn
from models.Flexible_CNN import Flexible_CNN
from PKLDataset import PKLDataset
from torch.utils.data import DataLoader



train_dataset = PKLDataset('../datasets/source/train/DC_T197_RP.txt')
val_dataset = PKLDataset('../datasets/source/validation/DC_T197_RP.txt')
out_path = "../datasets/info"
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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=3
)
