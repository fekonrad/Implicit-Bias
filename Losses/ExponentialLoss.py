import torch
import torch.nn as nn


class ExponentialLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(ExponentialLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        loss = torch.exp(-self.alpha * (y_true * y_pred))
        return loss.mean()
