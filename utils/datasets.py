import numpy as np
import torch
from torch.utils.data import Dataset


class LinearlySeparableDataset(Dataset):
    def __init__(self, num_samples=100, dim=2, random_seed=42):
        super(LinearlySeparableDataset, self).__init__()
        self.data = 10*(torch.rand((num_samples, dim))-0.5)
        self.slope = torch.rand(dim)

        self.labels = 2*((self.data @ self.slope) > 0).float()-1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]),  torch.tensor(self.labels[idx])


class TwoCircles(Dataset):
    def __init__(self, num_samples=100, dim=2, random_seed=42):
        self.data, self.labels = self.__create_circles__(num_samples, dim)

    def __create_circles__(self, num_samples, dim):
        x = np.zeros((num_samples,dim))
        y = np.zeros(num_samples)

        x[0][0] = 1.0
        y[0] = 1
        x[1][0] = -1.0
        y[1] = -1

        for i in range(2, num_samples):
          x[i] = np.random.normal(loc=0.0, scale=1.0, size=dim)
          x[i] = x[i]/np.linalg.norm(x[i])
          y[i] = 2*np.random.randint(0,2)-1

          if(y[i]>0):
            x[i][0] += 2.0
          else:
            x[i][0] -= 2.0
        return x , y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32),  torch.tensor(self.labels[idx], dtype=torch.float32)
