import torch
import os
from tqdm import tqdm


def pseudo_train_model(model, pseudo_loader,optimizer, criterion, device,
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

        for inputs, labels in tqdm(pseudo_loader, desc=f"Epoch {epoch+1} Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(f"Patience Counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience and epoch > num_epochs * 0.4:
                print(f"Early stopping at epoch {epoch + 1}.")
                break


    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(out_path, 'best_model.pth'))
        model.load_state_dict(best_model_state)

    return model,best_val_loss
def pseudo_test_model(model, criterion, pseudo_test_loader, device):
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val_samples = 0

    with torch.no_grad():
        for inputs, labels in pseudo_test_loader:
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