import torch
import torch.nn as nn
from models.Flexible_CNN import Flexible_CNN_FeatureExtractor, Flexible_CNN_Classifier

class Flexible_ADDA(nn.Module):
    def __init__(self, num_layers=2, start_channels=8, kernel_size=3,
                 cnn_act='leakrelu', num_classes=10):
        """
            Flexible_ADDA

            Feature extractor + classifier backbone used in ADDA-style domain adaptation.

            Args
            ----------
            num_layers : int, default 2
                Number of convolutional blocks in the CNN feature extractor.
            start_channels : int, default 8
                Number of channels in the first conv layer; later layers usually scale from this.
            kernel_size : int, default 3
                Convolution kernel size for all conv layers.
            cnn_act : str, default 'leakrelu'
                Name of activation function used in the CNN feature extractor.
            num_classes : int, default 10
                Number of output classes for classification.

            Forward Outputs
            ----------------------------
            class_outputs : Tensor
                Logits for class prediction, shape [B, num_classes].
            flat_feat : Tensor
                High-dimensional features directly from the CNN backbone, shape [B, feature_dim].
            reduced_features : Tensor
                Dimension-reduced features (e.g. for discriminator or metric losses), shape [B, 256].
            """
        super().__init__()
        self.feature_extractor = Flexible_CNN_FeatureExtractor(num_layers, start_channels, kernel_size, cnn_act)
        feature_dim = self.feature_extractor.feature_dim
        self.classifier = Flexible_CNN_Classifier(feature_dim, num_classes)
        self.feature_reducer = nn.Sequential(
            nn.Linear(feature_dim, 256, bias=False),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=0.1),
        )
        for m in self.feature_reducer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        flat_feat = self.feature_extractor(x)
        class_outputs = self.classifier(flat_feat)
        reduced_features = self.feature_reducer(flat_feat)  # [B, C]
        return class_outputs, flat_feat, reduced_features


class DomainClassifier(nn.Module):
    """
        DomainClassifier

        Simple MLP-based domain discriminator for adversarial or ADDA-style training.

        Args
        ----------
        feature_dim : int
            Dimensionality of the input feature vector.
        hidden : int, default 256
            Size of the hidden layer in the MLP.
        domain_dropout : float, default 0.2
            Dropout rate applied in the discriminator for regularization.
        num_domains : int, default 2
            Number of domain labels (e.g. 2 for source vs target).

        Forward Outputs
        ----------------------------
        logits : Tensor
            Domain logits of shape [B, num_domains], suitable for CrossEntropyLoss.
        """
    def __init__(self, feature_dim, hidden=256, domain_dropout=0.2, num_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(domain_dropout),
            nn.Linear(hidden, num_domains,bias=True)
        )
    def forward(self, x):
        return self.net(x)


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True