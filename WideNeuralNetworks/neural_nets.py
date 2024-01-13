import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.init as init
import torch.optim as optim


class TwoLayerReLU(nn.Module):
    def __init__(self, d=1, n=1000):  # d - input dimension ; n - number of neurons in hidden layer
        super().__init__()

        self.architecture = nn.Sequential(
            nn.Linear(d, n),
            nn.ReLU(),
            nn.Linear(n, 1)
        )
        scaling1 = np.sqrt(1 / d)
        scaling2 = np.sqrt(1 / n)

        init.uniform_(self.architecture[0].weight, -scaling1, scaling1)
        init.uniform_(self.architecture[0].bias, -scaling1, scaling1)

        init.uniform_(self.architecture[2].weight, -scaling2, scaling2)
        init.uniform_(self.architecture[2].bias, -scaling2, scaling2)

    def forward(self, x):
        return self.architecture(x)


class TwoLayerReLU_ASI(nn.Module):
    """
    two layer neural network with ReLU activation function and ASI initialization
    note that this creates a network with 2n hidden neurons, not n hidden neurons
    """
    def __init__(self, d=1, n=1000):  # d - input dimension ; n - number of neurons in hidden layer
        super().__init__()

        self.features1 = nn.Sequential(
            nn.Linear(d, n),
            nn.ReLU(),
            nn.Linear(n, 1)
        )

        self.features2 = nn.Sequential(
            nn.Linear(d, n),
            nn.ReLU(),
            nn.Linear(n, 1)
        )

        scaling1 = np.sqrt(1 / d)
        scaling2 = np.sqrt(1 / n)

        # initializing parameters with scaling as in (2)
        init.uniform_(self.architecture1[0].weight, -scaling1, scaling1)
        init.uniform_(self.architecture1[0].bias, -scaling1, scaling1)

        init.uniform_(self.architecture1[2].weight, -scaling2, scaling2)
        init.uniform_(self.architecture1[2].bias, -scaling2, scaling2)

        # clone parameters
        self.features2[0].weight.data = self.features1[0].weight.data.clone()
        self.features2[0].bias.data = self.features1[0].bias.data.clone()
        self.features2[2].weight.data = self.features1[2].weight.data.clone()
        self.features2[2].bias.data = self.features1[2].bias.data.clone()

    def forward(self, x):
        return (np.sqrt(2)/2)*(self.features1(x) - self.features2(x))

