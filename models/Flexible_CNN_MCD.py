import torch
import torch.nn as nn
import torch.nn.functional as F




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
    def __init__(self, num_layers=6, start_channels=8,kernel_size=15, cnn_act='leakrelu',input_size=2800):
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
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,bias=False))
            layers.append(nn.GroupNorm(num_groups=out_channels // 2, num_channels=out_channels))
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
    Flexible CNN classifier head with LayerNorm + Cosine similarity classifier.

    Args:
        feature_dim (int): Input feature dimension from CNN backbone.
        num_classes (int): Number of output classes.
        hidden (int): Hidden layer dimension.
        p (float): Dropout probability.
        temperature (float): Scaling factor for cosine logits.
    """
    def __init__(self, feature_dim, num_classes=10, hidden=256, p=0.2, temperature=0.05):
        super().__init__()
        self.temperature = temperature

        #  Feature projection layer: Linear -> LayerNorm -> activation -> Dropout
        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p)
        )

        # Weights of the cosine classifier
        self.weight = nn.Parameter(torch.Tensor(num_classes, hidden))
        nn.init.xavier_normal_(self.weight)  # 初始化

    def forward(self, x, return_feat=False):
        # Project features
        z = self.feat_proj(x)

        # Cosine similarity classifier
        z_norm = F.normalize(z, dim=1)          # Feature normalization
        w_norm = F.normalize(self.weight, dim=1) # Weight normalization
        logits = (z_norm @ w_norm.t()) / self.temperature

        if return_feat:
            return logits, z
        return logits

class Flexible_MCD(nn.Module):
    def __init__(self, num_layers=6, start_channels=8, kernel_size=15, cnn_act='leakrelu',
                 num_classes=10, input_size=2800, hidden=512, p=0.4, temperature=0.05):
        super().__init__()
        self.feature_extractor = Flexible_CNN_FeatureExtractor(
            num_layers=num_layers, start_channels=start_channels,
            kernel_size=kernel_size, cnn_act=cnn_act, input_size=input_size
        )
        feature_dim = self.feature_extractor.feature_dim
        self.c1 = Flexible_CNN_Classifier(feature_dim, num_classes=num_classes, hidden=hidden, p=p, temperature=temperature)
        self.c2 = Flexible_CNN_Classifier(feature_dim, num_classes=num_classes, hidden=hidden, p=p, temperature=temperature)
        self.feature_reducer = nn.Sequential(
            nn.Linear(feature_dim, 512, bias=False),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=0.1),
        )
        for m in self.feature_reducer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        features = self.feature_extractor(x)
        reduced_features = self.feature_reducer(features)
        l1 = self.c1(features)
        l2 = self.c2(features)
        return l1, l2, reduced_features