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

        A simple MLP-based domain discriminator used in DANN / adversarial domain
        adaptation setups. It takes feature vectors from the shared backbone and
        predicts a domain label (e.g. source vs target).

        Parameters
        ----------
        feature_dim : int
            Dimensionality of the input feature vector.
        hidden : int, default 256
            Size of the hidden layer.
        domain_dropout : float, default 0.2
            Dropout rate applied to the hidden representation for regularization.
        num_domains : int, default 2
            Number of domain classes (e.g. 2 for source and target).

        Forward
        -------
        x : Tensor, shape [B, feature_dim]
            Input feature tensor.

        Returns
        -------
        Tensor
            Domain logits of shape [B, num_domains], typically used with
            CrossEntropyLoss.
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


class Flexible_DANN(nn.Module):
    """
        Flexible_DANN

        A DANN-style domain-adversarial network built on top of a configurable
        CNN feature extractor. The model provides:
          - feature_extractor: shared CNN backbone producing flattened features
          - classifier:        label classifier trained on the source domain
          - domain_classifier: domain discriminator for adversarial alignment
          - feature_reducer:   learnable projection of features to a lower-dim space
                               (e.g. for additional regularizers / metrics)

        During training, the classifier is optimized to perform well on the
        source labels, while the domain classifier (optionally preceded by a
        Gradient Reversal Layer) encourages domain-invariant features.

        Parameters
        ----------
        num_layers : int, default 2
            Number of convolutional blocks in the CNN feature extractor.
        start_channels : int, default 8
            Number of channels in the first convolutional layer.
        kernel_size : int, default 3
            Kernel size used in the convolutional layers.
        cnn_act : str, default 'leakrelu'
            Activation type used in the CNN feature extractor.
        num_classes : int, default 10
            Number of label classes for the main classification task.
        lambda_ : float, default 1.0
            Initial coefficient for the Gradient Reversal Layer. This value can
            be updated externally (e.g. via a scheduling function) before each
            forward pass.

        Attributes
        ----------
        feature_extractor : nn.Module
            CNN backbone that outputs flattened feature vectors.
        classifier : nn.Module
            Linear / MLP head producing class logits.
        domain_classifier : DomainClassifier
            Domain discriminator operating on (optionally reversed) features.
        feature_reducer : nn.Sequential
            Projection network mapping features to a 256-dimensional space with
            LayerNorm, LeakyReLU, and dropout.
        lambda_ : float
            Current GRL coefficient used in the forward pass when grl=True.

        Forward
        -------
        x : Tensor
            Input batch, shape depends on Flexible_CNN_FeatureExtractor.
        grl : bool, default True
            If True, applies gradient reversal to the flat features before the
            domain classifier. If False, uses the raw flat features.

        Returns
        -------
        class_outputs : Tensor
            Class logits of shape [B, num_classes].
        domain_outputs : Tensor
            Domain logits of shape [B, num_domains].
        reduced_features : Tensor
            Reduced feature representation of shape [B, 256], suitable for
            additional alignment losses or analysis.
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
            nn.Dropout(p=0.1),
        )
        for m in self.feature_reducer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)



    def forward(self, x, grl=True):
        flat_feat = self.feature_extractor(x)
        reduced_features = self.feature_reducer(flat_feat)
        class_outputs = self.classifier(flat_feat)
        if grl:
            reversed_features = grad_reverse(flat_feat, self.lambda_)
        else:
            reversed_features = flat_feat
        domain_outputs = self.domain_classifier(reversed_features)

        return class_outputs, domain_outputs, reduced_features