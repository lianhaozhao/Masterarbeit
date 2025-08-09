import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna


def hyper_train_model(model, train_loader, val_loader, optimizer, criterion, device,
                num_epochs=20, early_stopping_patience=3,scheduler=None,trial = None):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if scheduler is not None:
            scheduler.step(val_loss)
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience Counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience and epoch > num_epochs*0.4:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    return best_val_loss

