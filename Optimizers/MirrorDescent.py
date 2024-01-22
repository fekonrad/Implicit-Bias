import torch
from torch.optim import Optimizer


class pGradientDescent(Optimizer):
  """
      Implementation from https://arxiv.org/pdf/2306.13853v1.pdf Appendix H  
  """
  def __init__(self, params, p=2.0, lr=1e-3):
    if p <= 1:
      raise ValueError(f"Invalid p: {p}, should be larger than 1")

    self.p = p
    defaults = dict(lr=lr, p=p)
    super(pGradientDescent, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(pGradientDescent, self).__setstate__(state)

  def step(self, closure=None):
    loss = None
    if closure is not None :
      with torch.enable_grad() :
        loss = closure()

    for group in self.param_groups:
      lr = group["lr"]
      p = group["p"]

      for param in group["params"]:
        if param.grad is None :
          continue

        x , dx = param.data , param.grad.data

        update = torch.pow(torch.abs(x), p-1) * torch.sign(x) - lr * dx
        param.data = torch.sign(update) * torch.pow (torch.abs(update), 1/(p-1))
    return loss

  
