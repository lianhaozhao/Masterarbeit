import torch
import torch.nn.functional as F


def generate_pseudo_labels(model, target_loader, device, threshold=0.9):
    """
    Generate pseudo-labels from unlabeled target data based on model predictions.

    Parameters:
        model : Trained model used for inference.
        target_loader (DataLoader): Dataloader providing unlabeled target domain data.
        threshold (float): Confidence threshold for accepting a pseudo-label. Default is 0.9.

    Returns:
        pseudo_data (Tensor): Selected input samples with confidence >= threshold.
        pseudo_labels (Tensor): Corresponding pseudo-labels predicted by the model.
    """

    model.eval()  # Set model to evaluation mode
    pseudo_data = []
    pseudo_labels = []

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for inputs in target_loader:
            # Handle (input, label) or other tuple/list format
            if isinstance(inputs, (tuple, list)):
                inputs = inputs[0]
            else:
                inputs = inputs

            inputs = inputs.to(device)
            outputs = model(inputs)  # Forward pass
            outputs = outputs[0]

            probs = F.softmax(outputs, dim=1)  # Convert logits to probabilities

            # Get the max probability (confidence) and the predicted class
            confidence, predicted = torch.max(probs, dim=1)

            # Select samples where confidence >= threshold
            mask = confidence >= threshold
            if mask.sum() > 0:
                selected_inputs = inputs[mask]
                selected_labels = predicted[mask]

                pseudo_data.append(selected_inputs.cpu())  # Move to CPU for later use
                pseudo_labels.append(selected_labels.cpu())

    # Concatenate all selected data and labels across batches
    if pseudo_data and pseudo_labels:
        pseudo_data = torch.cat(pseudo_data, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)
    else:
        pseudo_data = torch.empty(0)
        pseudo_labels = torch.empty(0)



    return pseudo_data, pseudo_labels



def generate_soft_pseudo_labels(model, target_loader, device, threshold=0.9):
    """
    Generate soft pseudo-labels for unlabeled target data using model predictions.

    Parameters:
        model (nn.Module): Trained model for generating pseudo-labels.
        target_loader (DataLoader): Dataloader for unlabeled target domain data.
        threshold (float): Confidence threshold for accepting a pseudo-label.
    Returns:
        pseudo_data (Tensor): Inputs with confidence above threshold.
        pseudo_probs (Tensor): Corresponding soft labels (probability distributions).
    """
    model.eval()
    pseudo_data = []
    pseudo_probs = []

    with torch.no_grad():
        for inputs in target_loader:
            if isinstance(inputs, (tuple, list)):
                inputs = inputs[0]

            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)  # Get soft labels

            confidence, _ = torch.max(probs, dim=1)
            mask = confidence >= threshold

            if mask.any():
                pseudo_data.append(inputs[mask].cpu())
                pseudo_probs.append(probs[mask].cpu())

    if pseudo_data and pseudo_probs:
        pseudo_data = torch.cat(pseudo_data, dim=0)
        pseudo_probs = torch.cat(pseudo_probs, dim=0)
    else:
        pseudo_data = torch.empty(0)
        pseudo_probs = torch.empty(0)

    print(f"Number of pseudo-labeled samples (soft): {len(pseudo_data)}")
    return pseudo_data, pseudo_probs
