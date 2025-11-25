import torch
import torch.nn as nn
from models.Flexible_CNN import Flexible_CNN_FeatureExtractor, Flexible_CNN_Classifier

class GradientReversalFunction(torch.autograd.Function):
    """
        GradientReversalFunction

        Autograd function that implements the Gradient Reversal Layer (GRL)
        used in DANN-style domain adaptation.

        Forward:
            Returns the input as-is.

        Backward:
            Multiplies the incoming gradient by -lambda_, effectively reversing
            the gradient direction and scaling it.

        Parameters
        ----------
        x : Tensor
            Input feature tensor.
        lambda_ : float
            Gradient reversal coefficient. Often scheduled from 0 to a maximum
            value (e.g. 0.5 or 1.0) during training.
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
        Apply a Gradient Reversal Layer (GRL) to the input tensor.

        This is a convenience wrapper around GradientReversalFunction.apply.

        Parameters
        ----------
        x : Tensor
            Input feature tensor.
        lambda_ : float, default 1.0
            Gradient reversal coefficient. Controls the strength of the
            adversarial signal from the domain classifier.

        Returns
        -------
        Tensor
            Tensor with the same values as x, but whose gradient is multiplied
            by -lambda_ during backpropagation.
        """
    return GradientReversalFunction.apply(x, lambda_)

class DomainClassifier(nn.Module):
    """
        DomainClassifier

        A simple MLP domain discriminator used in DANN. It receives features
        from the shared feature extractor (optionally passed through a GRL)
        and predicts the domain label (e.g. source vs target).

        Parameters
        ----------
        feature_dim : int
            Dimensionality of the input feature vector.
        hidden : int, default 256
            Size of the hidden layer.
        domain_dropout : float, default 0.2
            Dropout rate applied in the hidden layer for regularization.
        num_domains : int, default 2
            Number of domain classes (e.g. 2 for source and target).

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
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(domain_dropout),
            nn.Linear(hidden, num_domains)
        )
    def forward(self, x):
        return self.net(x)


class Flexible_DANN(nn.Module):
    """
      Flexible_DANN

      A domain-adversarial neural network (DANN) built on top of a configurable
      CNN feature extractor. The model consists of:
        - feature_extractor: shared CNN backbone producing latent features
        - classifier:        task classifier head (trained on source labels)
        - domain_classifier: domain discriminator coupled with a GRL

      During training, the classifier is optimized for good source-domain
      performance, while the domain classifier tries to distinguish source
      from target. The GRL inverts the gradient from the domain classifier,
      encouraging the backbone to learn domain-invariant features.

      Parameters
      ----------
      num_layers : int, default 2
          Number of convolutional blocks in the CNN feature extractor.
      start_channels : int, default 8
          Number of channels in the first convolutional layer.
      kernel_size : int, default 3
          Convolution kernel size used in the feature extractor.
      cnn_act : str, default 'leakrelu'
          Activation type used in the CNN feature extractor.
      num_classes : int, default 10
          Number of label classes for the task classifier.
      lambda_ : float, default 1.0
          Initial gradient reversal coefficient for the GRL. Can be updated
          externally (e.g. by a scheduling function) before each forward pass.

      Forward
      -------
      x : Tensor
          Input batch, expected shape depends on Flexible_CNN_FeatureExtractor.

      grl : bool, default True
          If True, applies gradient reversal before the domain classifier.
          If False, passes features directly to the domain classifier.

      Returns
      -------
      class_outputs : Tensor
          Classification logits of shape [B, num_classes].
      domain_outputs : Tensor
          Domain logits of shape [B, num_domains].
      """
    def __init__(self, num_layers=2, start_channels=8, kernel_size=3,
                 cnn_act='leakrelu', num_classes=10, lambda_=1.0):
        super().__init__()
        self.feature_extractor = Flexible_CNN_FeatureExtractor(num_layers, start_channels, kernel_size, cnn_act)
        feature_dim = self.feature_extractor.feature_dim
        self.classifier = Flexible_CNN_Classifier(feature_dim, num_classes)
        self.domain_classifier = DomainClassifier(feature_dim)
        self.lambda_ = lambda_

    def forward(self, x, grl=True):
        features = self.feature_extractor(x)
        class_outputs = self.classifier(features)
        if grl:
            reversed_features = grad_reverse(features, self.lambda_)
        else:
            reversed_features = features
        domain_outputs = self.domain_classifier(reversed_features)
        return class_outputs, domain_outputs