from typing import List

import torch
import torch.nn as nn
from torch.distributions import Categorical


def make_net(sizes: List[int],
             activation,
             output_activation=nn.Identity()) -> nn.Module:
    """
    Make torch neural network.

    :param sizes: List of net layer neural numbers.
    :param activation: Non-output activation.
    :param output_activation: Output activation.
    :return: Torch neural network.
    """
    layers = []
    for i in range(len(sizes)-1):
        active = activation if i < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), active]
    return nn.Sequential(*layers)


class MLPRegression(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_sizes=(64, 64), activation=nn.Tanh()):
        """
            @param int - input_dim
            @param int - output_dim
            @param list - hidden_sizes : such as [32,32,32]
        """
        super().__init__()
        self.net = make_net([input_dim] + list(hidden_sizes) + [output_dim], activation)

    def forward(self, x):
        """
            @param tensor - x: shape [batch, input dim]

            @return tensor - out : shape [batch, output dim]
        """
        out = self.net(x)
        return out


class MLPCategorical(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_sizes=(64,64), activation=nn.Tanh()):
        """
            @param int - input_dim
            @param int - output_dim
            @param list - hidden_sizes : such as [32,32,32]
        """
        super().__init__()
        self.logits_net = make_net([input_dim] + list(hidden_sizes) + [output_dim], activation)

    def forward(self, x):
        """
            @param tensor - x: shape [batch, input dim]

            @return tensor - out : shape [batch, 1]
        """
        logits = self.logits_net(x)
        out = Categorical(logits=logits)
        return torch.squeeze(out, -1)

