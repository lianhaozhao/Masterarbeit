import torch
import torch.nn as nn
from models.Flexible_CNN import Flexible_CNN_FeatureExtractor, Flexible_CNN_Classifier

class GradientReversalFunction(torch.autograd.Function):
    """
        Autograd function implementing a Gradient Reversal Layer (GRL).

        Forward:
            Returns the input tensor unchanged.

        Backward:
            Multiplies the incoming gradient by -lambda_, effectively reversing
            and scaling the gradient. This is typically used in domain-adversarial
            training (e.g., DANN) to encourage domain-invariant features.

        Parameters
        ----------
        x : Tensor
            Input feature tensor.
        lambda_ : float
            Gradient reversal coefficient. Common practice is to schedule this
            value from 0 to some maximum (e.g., 0.5 or 1.0) over the course of
            training.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        grad_x = -ctx.lambda_ * grad_output
        return grad_x, None


def grad_reverse(x, lambda_=1.0):
    """
        Convenience wrapper for applying the Gradient Reversal Layer (GRL).

        Parameters
        ----------
        x : Tensor
            Input feature tensor.
        lambda_ : float, default 1.0
            Gradient reversal coefficient that scales the reversed gradient.

        Returns
        -------
        Tensor
            Tensor with identical forward values as x, but with gradients
            multiplied by -lambda_ during backpropagation.
    """
    return GradientReversalFunction.apply(x, lambda_)

class DomainClassifier(nn.Module):
    """
        A simple MLP-based domain discriminator for domain-adversarial training.

        This module receives feature vectors (optionally after a GRL) and predicts
        domain labels (e.g., source vs target), providing an adversarial signal to
        the shared feature extractor.

        Parameters
        ----------
        feature_dim : int
            Dimensionality of the input feature vector.
        hidden : int, default 256
            Size of the hidden layer.
        domain_dropout : float, default 0.2
            Dropout rate applied to the hidden representation for regularization.
        num_domains : int, default 2
            Number of domain classes (e.g., 2 for source and target).

        Forward
        -------
        x : Tensor, shape [B, feature_dim]
            Input features.

        Returns
        -------
        Tensor
            Domain logits of shape [B, num_domains], suitable for CrossEntropyLoss.
    """
    def __init__(self, feature_dim, hidden=256, domain_dropout=0.2, num_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden,bias=False),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(domain_dropout),
            nn.Linear(hidden, num_domains,bias=True),
        )
    def forward(self, x):
        return self.net(x)


class Flexible_DANN(nn.Module):
    """
    Flexible_DANN

    A configurable DANN-style (Domain-Adversarial Neural Network) model
    built on top of a CNN feature extractor. It includes:
      - feature_extractor: shared CNN backbone producing flattened features
      - classifier:        task classifier head for label prediction
      - domain_classifier: domain discriminator (with optional GRL in front)
      - feature_reducer:   projection of features into a lower-dimensional
                           space (e.g., for additional losses or analysis)

    Parameters
    ----------
    num_layers : int, default 2
        Number of convolutional blocks in the CNN feature extractor.
    start_channels : int, default 8
        Number of channels in the first convolutional layer.
    kernel_size : int, default 3
        Convolution kernel size in the feature extractor.
    cnn_act : str, default 'leakrelu'
        Activation used in the CNN feature extractor.
    num_classes : int, default 10
        Number of label classes for the task classifier.
    lambda_ : float, default 1.0
        Initial coefficient for the Gradient Reversal Layer. This value can
        be updated externally (e.g., by a scheduling function) during training.

    Forward
    -------
    x : Tensor
        Input batch, shape determined by Flexible_CNN_FeatureExtractor.
    grl : bool, default True
        If True, applies gradient reversal before the domain classifier.
        If False, passes features directly to the domain classifier.

    Returns
    -------
    class_outputs : Tensor
        Class logits of shape [B, num_classes].
    domain_outputs : Tensor
        Domain logits of shape [B, num_domains].
    reduced_features : Tensor
        Reduced feature representation of shape [B, 256], produced by
        feature_reducer, useful for downstream metrics or additional
        alignment losses.
    """
    def __init__(self, num_layers=2, start_channels=8, kernel_size=3,
                 cnn_act='leakrelu', num_classes=10, lambda_=1.0):
        super().__init__()
        self.feature_extractor = Flexible_CNN_FeatureExtractor(num_layers, start_channels, kernel_size, cnn_act)
        feature_dim = self.feature_extractor.feature_dim
        self.classifier = Flexible_CNN_Classifier(feature_dim, num_classes)
        self.domain_classifier = DomainClassifier(feature_dim)
        self.lambda_ = lambda_
        self.feature_reducer = nn.Sequential(
            nn.Linear(feature_dim, 256, bias=False),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(p=0.1) ,
        )
        for m in self.feature_reducer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)



    def forward(self, x, grl=True):
        features = self.feature_extractor(x)
        class_outputs = self.classifier(features)
        if grl:
            reversed_features = grad_reverse(features, self.lambda_)
        else:
            reversed_features = features
        domain_outputs = self.domain_classifier(reversed_features)
        reduced_features = self.feature_reducer(features)
        return class_outputs, domain_outputs, reduced_features