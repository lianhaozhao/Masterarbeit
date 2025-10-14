import torch
import torch.nn as nn
from models.Flexible_CNN import Flexible_CNN_FeatureExtractor, Flexible_CNN_Classifier

class Flexible_ADDA(nn.Module):
    def __init__(self, num_layers=2, start_channels=8, kernel_size=3,
                 cnn_act='leakrelu', num_classes=10):
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

    def forward(self, x):
        conv_out, flat_feat = self.feature_extractor(x, return_conv=True)
        class_outputs = self.classifier(flat_feat)
        reduced_features = self.feature_reducer(conv_out)  # [B, C]
        return class_outputs, flat_feat, reduced_features


class DomainClassifier(nn.Module):
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