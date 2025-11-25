import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
import copy
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pandas as pd
def pseudo_train_model(model, pseudo_loader,optimizer, criterion, device,
                num_epochs=20, early_stopping_patience=3, scheduler=None, out_path=None):
    """
        Trains a model using pseudo-labeled data with optional early stopping and learning rate scheduling.

        Args:
            model (nn.Module): The model to be trained.
            pseudo_loader (DataLoader): DataLoader containing pseudo-labeled samples (inputs:no label).
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
            device (torch.device): Device to run training on ('cuda' or 'cpu').
            num_epochs (int): Maximum number of training epochs.
            early_stopping_patience (int): Stop training early if loss does not improve for this many epochs.
            scheduler : Learning rate scheduler (e.g., ReduceLROnPlateau).
            out_path (str): Directory to save the best model checkpoint.

        Returns:
            model (nn.Module): The trained model with best weights loaded.
            best_val_loss (float): Lowest training loss achieved during training.
        """

    best_train_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        for inputs, labels in tqdm(pseudo_loader, desc=f"Epoch {epoch+1} Training"):
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

        if scheduler is not None:
            scheduler.step(train_loss)


        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"- Train Loss: {train_loss:.6f}, Acc: {train_accuracy:.4f} "
             )

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            print(f"Patience Counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience and epoch > num_epochs * 0.3:
                print(f"Early stopping at epoch {epoch + 1}.")
                break


    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(out_path, 'best_model.pth'))
        model.load_state_dict(best_model_state)

    return model,best_train_loss
def general_test_model(model, criterion, general_test_loader, device):
    """
        Evaluates the model on a labeled test set using standard classification metrics.

        Args:
            model (nn.Module): Trained model to evaluate.
            criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
            general_test_loader (DataLoader): DataLoader containing test data with ground truth labels.
            device (torch.device): Device for evaluation ('cuda' or 'cpu').

        Prints:
            Test loss and accuracy.
        """
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val_samples = 0

    with torch.no_grad():
        for inputs, labels in general_test_loader:
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
    print(f"- test Loss: {val_loss:.6f}, test Acc: {val_accuracy:.4f}")

def pseudo_soft_train_model(model, pseudo_loader,optimizer, device,
                num_epochs=20, early_stopping_patience=3, scheduler=None, out_path=None):
    """
        Trains a model using pseudo-labeled data with optional early stopping and learning rate scheduling.

        Args:
            model (nn.Module): The model to be trained.
            pseudo_loader (DataLoader): DataLoader containing pseudo-labeled samples (inputs:no label).
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
            device (torch.device): Device to run training on ('cuda' or 'cpu').
            num_epochs (int): Maximum number of training epochs.
            early_stopping_patience (int): Stop training early if loss does not improve for this many epochs.
            scheduler : Learning rate scheduler (e.g., ReduceLROnPlateau).
            out_path (str): Directory to save the best model checkpoint.

        Returns:
            model (nn.Module): The trained model with best weights loaded.
            best_val_loss (float): Lowest training loss achieved during training.
        """

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        for inputs, labels in tqdm(pseudo_loader, desc=f"Epoch {epoch+1} Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            log_probs = F.log_softmax(outputs, dim=1)
            loss = F.kl_div(log_probs, labels, reduction='batchmean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = inputs.size(0)
            train_loss += loss.item() * batch_size_actual
            total_train_samples += batch_size_actual

            # Accuracy calculation (training)
            _, preds = torch.max(outputs, dim=1)
            _, true_labels = torch.max(labels, dim=1)  # 从 soft label 得到类索引
            correct_train += (preds == true_labels).sum().item()

        train_loss /= total_train_samples
        train_accuracy = correct_train / total_train_samples

        if scheduler is not None:
            scheduler.step(train_loss)


        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"- Train Loss: {train_loss:.6f}, Acc: {train_accuracy:.4f} "
             )

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            print(f"Patience Counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience and epoch > num_epochs * 0.3:
                print(f"Early stopping at epoch {epoch + 1}.")
                break


    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(out_path, 'best_model.pth'))
        model.load_state_dict(best_model_state)

    return model,best_val_loss


def pseudo_soft_test_model(model, criterion, pseudo_test_loader, device, use_soft_labels=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in pseudo_test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs

            if use_soft_labels:
                # 确保targets是浮点型概率分布
                targets = targets.float()  # 新增强制类型转换
                log_probs = F.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, targets)
            else:
                # 硬标签处理保持不变
                loss = criterion(outputs, targets.long() if targets.dtype != torch.long else targets)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    if use_soft_labels:
        print(f"- Test KL Divergence: {avg_loss:.6f}")
    else:
        print(f"- Test Loss: {avg_loss:.6f}, Acc: {correct / total_samples:.4f}")
def general_test_model_plot(model, criterion, general_test_loader, device,
                            save_fig=False, fig_path="./confusion_matrix.png",
                            save_table=True, table_path="./test_stats.csv",
                            save_cm_table=True, cm_table_path="./confusion_matrix.csv"):
    """
    Evaluates the model on a labeled test set, plots and saves raw (non-normalized) confusion matrix,
    and saves classification statistics.

    Returns:
        val_loss, val_accuracy, cm (raw confusion matrix), df_report, df_cm_raw (pandas DataFrame)
    """
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val_samples = 0
    preds_list, trues_list = [], []

    with torch.no_grad():
        for inputs, labels in general_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(outputs, labels)

            batch_size_actual = inputs.size(0)
            val_loss += loss.item() * batch_size_actual
            total_val_samples += batch_size_actual

            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()

            preds_list.append(preds.cpu().numpy())
            trues_list.append(labels.cpu().numpy())

    preds_all = np.concatenate(preds_list)
    trues_all = np.concatenate(trues_list)

    val_loss /= total_val_samples
    val_accuracy = correct_val / total_val_samples
    print(f"- test Loss: {val_loss:.6f}, test Acc: {val_accuracy:.4f}")

    # ======== Confusion matrix (raw count) ========
    class_labels = ['R05', 'R10', 'R15', 'R20', 'R25', 'R30', 'R35', 'R40', 'R45', 'R50']
    cm = confusion_matrix(trues_all, preds_all, labels=np.arange(len(class_labels)))

    # ======== draw ========
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues', xticks_rotation=45, values_format="d")
    plt.title(f"Confusion Matrix (Raw Counts) | Acc={val_accuracy:.4f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    if save_fig:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # ======== Save the confusion matrix table ========
    df_cm_raw = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    if save_cm_table:
        os.makedirs(os.path.dirname(cm_table_path), exist_ok=True)
        df_cm_raw.to_csv(cm_table_path, float_format="%.0f")

    # ======== Classification Report ========
    report_dict = classification_report(
        trues_all, preds_all, target_names=class_labels, output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report_dict).transpose()

    if save_table:
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        df_report.to_csv(table_path, float_format="%.4f")

    return val_loss, val_accuracy, cm, df_report, df_cm_raw
