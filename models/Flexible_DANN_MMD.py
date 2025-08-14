import torch
import torch.nn as nn
from models.Flexible_CNN import Flexible_CNN_FeatureExtractor, Flexible_CNN_Classifier

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        grad_x = -ctx.lambda_ * grad_output
        return grad_x, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

class DomainClassifier(nn.Module):
    def __init__(self, feature_dim, hidden=256, domain_dropout=0.2, num_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(domain_dropout),
            nn.Linear(hidden, num_domains)
        )
    def forward(self, x):
        return self.net(x)


class Flexible_DANN(nn.Module):
    def __init__(self, num_layers=2, start_channels=8, kernel_size=3,
                 cnn_act='leakrelu', num_classes=10, lambda_=1.0):
        super().__init__()
        self.feature_extractor = Flexible_CNN_FeatureExtractor(num_layers, start_channels, kernel_size, cnn_act)
        feature_dim = self.feature_extractor.feature_dim
        self.classifier = Flexible_CNN_Classifier(feature_dim, num_classes)
        self.domain_classifier = DomainClassifier(feature_dim)
        self.lambda_ = lambda_
        # self.feature_reducer = nn.Sequential(
        #     nn.Linear(feature_dim, 512)
        # )

    def forward(self, x, grl=True):
        features = self.feature_extractor(x)
        class_outputs = self.classifier(features)
        if grl:
            reversed_features = grad_reverse(features, self.lambda_)
        else:
            reversed_features = features
        domain_outputs = self.domain_classifier(reversed_features)
        # reduced_features = self.feature_reducer(features)
        return class_outputs, domain_outputs, features