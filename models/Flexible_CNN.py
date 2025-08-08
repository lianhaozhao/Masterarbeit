import torch
import torch.nn as nn

class Flexible_CNN_FeatureExtractor(nn.Module):
    """
       A flexible 1D CNN feature extractor for time-series or signal data.

       Automatically adjusts output feature dimension based on input length.

       Args:
           num_layers (int): Number of convolutional layers.
           start_channels (int): Number of channels in the first conv layer.
           kernel_size (int): Kernel size for all conv layers.
           cnn_act (str): Activation function ('relu', 'leakrelu', 'sigmoid', 'tanh').
           input_size (int): Input sequence length (used to infer output shape).
       """
    def __init__(self, num_layers=2, start_channels=8,kernel_size=3, cnn_act='leakrelu',input_size=2800):
        super(Flexible_CNN_FeatureExtractor, self).__init__()

        activation_dict = {
            'relu': nn.ReLU,
            'leakrelu': nn.LeakyReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh
        }
        if cnn_act not in activation_dict:
            raise ValueError(f"Unsupported activation function: {cnn_act}")
        activation_fn = activation_dict[cnn_act]

        layers = []
        in_channels = 1
        for i in range(num_layers):
            out_channels = start_channels * (2 ** i)
            padding = (kernel_size - 1) // 2
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(activation_fn())
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size)  # B=1, C=1, L=input_size
            out = self.conv(dummy_input)
            self.feature_dim = out.shape[1] * out.shape[2]  # C × L

    def forward(self, x):
        x = self.conv(x)  # (B, C, 1)
        x = x.view(x.size(0), -1)  # 展平为 (B, C)
        return x

class Flexible_CNN_Classifier(nn.Module):
    """
       A simple classifier head using a fully connected layer.

       Args:
           feature_dim (int): Input feature dimension from the CNN.
           num_classes (int): Number of output classes.
       """
    def __init__(self, feature_dim, num_classes=10):
        super(Flexible_CNN_Classifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class Flexible_CNN(nn.Module):
    """
        A complete CNN model combining the feature extractor and classifier.

        Args:
            num_layers (int): Number of conv layers in the feature extractor.
            start_channels (int): Base number of conv channels.
            kernel_size (int): Kernel size for convolutions.
            cnn_act (str): Activation function name.
            num_classes (int): Number of output classes.
        """
    def __init__(self, num_layers=2, start_channels=8, kernel_size=3, cnn_act='leakrelu', num_classes=10):
        super(Flexible_CNN, self).__init__()
        self.feature_extractor = Flexible_CNN_FeatureExtractor(
            num_layers=num_layers,
            start_channels=start_channels,
            cnn_act=cnn_act,
            kernel_size=kernel_size
        )
        feature_dim = self.feature_extractor.feature_dim
        self.classifier = Flexible_CNN_Classifier(feature_dim, num_classes=num_classes)

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)       # (B, D)
        out = self.classifier(features)            # (B, num_classes)
        if return_features:
            return out, features
        else:
            return out


def freeze_feature_train_head(model, lr, weight_decay):
    """
        Freezes the feature extractor and returns an optimizer for training only the classifier head.

        Args:
            model (nn.Module): The model with 'feature_extractor' and 'classifier' attributes.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 regularization).

        Returns:
            torch.optim.Optimizer: Optimizer for the classifier parameters only.
        """
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer




