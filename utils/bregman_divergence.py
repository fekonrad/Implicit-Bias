import numpy as np


def bregman_div(x, y, p):
  # assumes x and y to be numpy arrays of the same shape
  # computes the bregman divergence D_\psi(x,y) for \psi(z)=p^{-1} |z|_p^p
  # compute nabla psi (y):
  dim = x.shape[0]
  nabla_psi_y = np.zeros(dim)
  for i in range(dim):
    nabla_psi_y[i] = np.sign(y[i]) * np.power(np.abs(y[i]), p-1)
  return np.linalg.norm(x, ord=p)/p - np.linalg.norm(y, ord=p)/p - np.dot(nabla_psi_y, x-y)
