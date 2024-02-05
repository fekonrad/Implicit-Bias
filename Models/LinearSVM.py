import torch
from torch import nn


class LinearSVM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, 1, bias=False)            # homogeneous model, no bias

    def forward(self, x):
        return self.net(x)


class ExponentialLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(ExponentialLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        loss = torch.exp(-self.alpha * (y_true * y_pred))
        return loss.mean()


