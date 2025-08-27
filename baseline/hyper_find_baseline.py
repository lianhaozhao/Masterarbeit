import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from torch.utils.data import DataLoader
from models.Flexible_CNN import Flexible_CNN
from PKLDataset import PKLDataset
from utils.train_utils import hyper_train_model
import json


def hyper_optimization(trial):
    """
        Execute a single Optuna hyperparameter optimization trial.

        This function defines the hyperparameter search space, builds
        the training and validation DataLoaders, initializes the Flexible_CNN
        model along with its optimizer and scheduler, runs the training process,
        and saves the results to a JSON file after each trial.

        Args:
            trial (optuna.trial.Trial):
                An Optuna Trial object used for sampling hyperparameters
                and reporting intermediate results.

        Returns:
            float:
                The best validation loss (best_val_loss) achieved in this trial.

        Hyperparameters being searched:
            - batch_size (int):
                One of {16, 32, 64}; controls the number of samples per training step.
            - learning_rate (float):
                Sampled log-uniformly in [1e-5, 5e-3].
            - weight_decay (float):
                L2 regularization coefficient, sampled log-uniformly in [1e-4, 1e-3].
            - num_layers (int):
                The number of convolutional layers, between 3 and 7.
            - kernel_size (int):
                Size of the convolution kernel, chosen from {3, 5, 7, 15, 31}.
            - start_channels (int):
                The number of output channels in the first convolutional layer,
                chosen from {4, 8}.

        Workflow:
            1. Sample hyperparameters from the given trial.
            2. Build the training and validation DataLoaders.
            3. Initialize the Flexible_CNN model.
            4. Define the loss function (CrossEntropyLoss), optimizer (Adam),
               and scheduler (ReduceLROnPlateau).
            5. Train the model using hyper_train_model with early stopping
               and learning rate scheduling.
            6. Save the trial hyperparameters and best_val_loss to
               ../datasets/info/params.json.
            7. Return the best_val_loss for this trial.

        Note:
            - The ReduceLROnPlateau scheduler adjusts the learning rate dynamically
              based on validation loss, with factor fixed at 0.7 and patience at 3.
    """

    batch_size = trial.suggest_categorical("batch_size", [32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True)
    num_layers = trial.suggest_int("num_layers", 3, 7)
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 15, 31])
    start_channels = trial.suggest_categorical("start_channels", [4, 8])

    print(f"Trial {trial.number}: batch_size={batch_size}, lr={learning_rate}, wd={weight_decay}, "
          f"layers={num_layers}, channels={start_channels}, kernel_size={kernel_size}")

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

    best_val_loss= hyper_train_model(
        model, train_loader, val_loader, optimizer, criterion, device,
        num_epochs=30, early_stopping_patience=3,scheduler=scheduler,trial = trial
    )

    # 保存超参数
    trial_params = {
        "trial": trial.number,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_layers": num_layers,
        "start_channels": start_channels,
        "kernel_size": kernel_size,
        "best_val_loss": best_val_loss,
    }
    out_path = "../datasets/info"
    trial_params_path = os.path.join(out_path, "params.json")
    # 如果文件存在，先读取再追加
    if os.path.exists(trial_params_path):
        with open(trial_params_path, "r") as f:
            all_trials = json.load(f)
    else:
        all_trials = []

    all_trials.append(trial_params)

    # 回写文件
    with open(trial_params_path, "w") as f:
        json.dump(all_trials, f, indent=4)

    return best_val_loss



if __name__ == '__main__':
    train_dataset = PKLDataset('../datasets/source/train/DC_T197_RP.txt')
    val_dataset = PKLDataset('../datasets/source/validation/DC_T197_RP.txt')
    out_path = "../datasets/info2"
    os.makedirs(out_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(hyper_optimization, n_trials=100 )
    best_result = {
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_trial.params,
        "best_value": study.best_trial.value
    }
    with open(os.path.join(out_path, "best_params.json"), "w") as json_file:
        json.dump(best_result, json_file, indent=4)
    print(f"Beste Versuch: {study.best_trial.number}")
    print(f"Beste Hyperparameter: {study.best_trial.params}")
    print(f"Bester Validierungsverlust: {study.best_value}")

