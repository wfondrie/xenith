"""
Define models that xenith can use.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Create a multilayer perceptron model with variable hidden layers.

    The network will have the specified number of layers and neurons,
    with each layer using ReLU activation.

    Parameters
    ----------
    input_dim : int
        The number of input features.

    hidden_dims : list of int
        The number of neurons in each hidden layer. Each element
        specifies a new hidden layer.
    """
    def __init__(self, input_dim, hidden_dims):
        """Instantiate an MLP object"""
        super(MLP, self).__init__()
        layers = hidden_dims # for consistency

        if not input_dim or not isinstance(input_dim, int):
            raise ValueError("'input_dim' must be a non-zero integer.")

        if isinstance(layers, int):
            layers = [layers]

        if not all(layers) or not all(isinstance(l, int) for l in layers):
            raise ValueError("'hidden_dims' must be a list of non-zero "
                             "integers.")

        layers = list(layers) # needed if layers is a Tuple.
        in_layers = [input_dim] + layers
        out_layers = layers + [1]
        for idx, in_layer in enumerate(in_layers):
            self.add_module("linear_{}".format(idx),
                            nn.Linear(in_layer, out_layers[idx]))

    def forward(self, x):
        """Define the forward pass"""
        for idx, layer in enumerate(self._modules.values()):
            if idx < len(self._modules)-1:
                x = F.relu_(layer(x))
            else:
                return layer(x)


class Linear(nn.Module):
    """
    Create a simple linear model.

    Parameters
    ----------
    input_dim : int
        The number of input features
    """
    def __init__(self, input_dim):
        """Instantiate a linear model"""
        if not input_dim or not isinstance(input_dim, int):
            raise ValueError("'input_dim' must be a non-zero integer.")

        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """The forward pass"""
        return self.linear(x)


# Define Sigmoid loss function ------------------------------------------------
class SigmoidLoss(nn.Module):
    """
    Create a sigmoid loss criterion.

    The sigmoid loss function takes the output score, :math:`S`, and the
    target, :math:`Y`, to calculate the mean loss, :math:`L`, using:

    ..math::
        L = Y*(1-\sigma(S)) + (1-Y)*(\sigma(S))

    where:

    ..math::
        \sigma(S) = \frac{1}{1+exp(-S)}
    """
    def __init__(self):
        """Initialize a SigmoidLoss object"""
        super(SigmoidLoss, self).__init__()

    def forward(self, score, target):
        """
        Calculate the loss using a sigmoid loss function.

        Parameters
        ----------
        score : torch.FloatTensor
            A 1D tensor of the scores output from a model. These should
            not be sigmoided.

        target : torch.ByteTensor
            A 1D tensor of indicating the truth. In the case of xenith
            '1' indicates a target hit and '0' indicates a decoy hit.
        """
        if not score.is_floating_point():
            score = score.float()

        if not target.is_floating_point():
            target = target.float()

        eps = torch.finfo(score.dtype).eps
        pred = torch.sigmoid(score).clamp(min=eps, max=1 - eps)
        loss = target * (1 - pred) + (1 - target) * pred
        return loss.mean()


class HybridLoss(nn.Module):
    """
    Create a hybrid loss criterion.

    The hybrid loss function uses a log loss form for Decoy labels and a
    sigmoid loss for Target labels. The hybrid loss function takes the
    output score, :math:`S`, and the target, :math:`Y`, to calculate the
    mean loss, :math:`L`, using:

    ..math::
        L = Y*(1 - \sigma(S)) - (1-Y)*log(1 - (\sigma(S)))

    where:

    ..math::
        \sigma(S) = \frac{1}{1+exp(-S)}
    """
    def __init__(self):
        """Initialize a SigmoidLoss object"""
        super(HybridLoss, self).__init__()

    def forward(self, score, target):
        """
        Calculate the loss using a sigmoid loss function.

        Parameters
        ----------
        score : torch.FloatTensor
            A 1D tensor of the scores output from a model. These should
            not be sigmoided.

        target : torch.ByteTensor
            A 1D tensor of indicating the truth. In the case of xenith
            '1' indicates a target hit and '0' indicates a decoy hit.
        """
        if not score.is_floating_point():
            score = score.float()

        if not target.is_floating_point():
            target = target.float()

        eps = torch.finfo(score.dtype).eps
        pred = torch.sigmoid(score).clamp(min=eps, max=1 - eps)
        loss = target * (1 - pred) - (1 - target) * torch.log(1 - pred)
        return loss.mean()
