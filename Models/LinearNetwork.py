import numpy as np
import torch
from torch import nn


class LinearNetwork(nn.Module):
  def __init__(self, dimensions):
    """
      - dimensions: list of integers representing the number of neurons in each hidden layer
                    dimensions[0] has to be equal to the input dimension
                    dimensions[-1] has to be equal to the output dimension
    """
    super().__init__()
    self.dimensions = dimensions
    self.layers = nn.ModuleList()
    for i in range(1, len(dimensions)):
        self.layers.append(nn.Linear(dimensions[i-1], dimensions[i], bias=False))   # homogeneous model, no bias

  def w(self):
    identity_tensor = torch.eye(self.dimensions[0], dtype=torch.float32, device=self.device)
    w_tens = self.forward(identity_tensor)
    return w_tens.detach().cpu().numpy().flatten()

  def get_layer(self, i):
    if i < 0 or i >= len(self.layers):
        raise ValueError("Invalid layer index")
    return self.layers[i].weight

  def parameter_norm(self, p, layer='all'):
    if layer == 'all':
          norm = 0.0
          for param in self.parameters():
              norm += torch.pow(torch.abs(param), p).sum()
          return np.power(norm.item(), 1/p)
    else:
      param = self.layers[layer].weight
      return np.power(torch.pow(torch.abs(param), p).sum().item(), 1/p)

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
