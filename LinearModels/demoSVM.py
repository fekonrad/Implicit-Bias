import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from LinearSVM import LinearSVM
from LinearSVM import ExponentialLoss

class LinearlySeparableDataset(Dataset):
    def __init__(self, num_samples=100, random_seed=42):
        super(LinearlySeparableDataset, self).__init__()
        self.data = 10*(torch.rand((num_samples, 2))-0.5)
        self.slope = torch.rand(1)

        self.labels = 2*((self.data[:, 1] - self.slope * self.data[:, 0]) > 0).float()-1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]),  torch.tensor(self.labels[idx])


def visualize_model(model, data_loader, title):
    model.eval()
    data, labels = next(iter(data_loader))

    data1 = data[labels == 1]
    data0 = data[labels == -1]
    plt.scatter(data1[:, 0], data1[:, 1], c='r', marker='o', edgecolors='k', s=50)
    plt.scatter(data0[:, 0], data0[:, 1], c='b', marker='x', edgecolors='k', s=50)
    plt.title(title)

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.tensor(mesh_data, dtype=torch.float32)

    with torch.no_grad():
        model.eval()
        predictions = torch.sign(model(mesh_tensor)).cpu().numpy()

    plt.contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.3, cmap='viridis')
    plt.show()


# Training ...
linearSVM = LinearSVM(dim=2)
linear_dataset = LinearlySeparableDataset(num_samples=10)
dataloader = DataLoader(linear_dataset, batch_size=20, shuffle=True)

# visualize at initialisation
visualize_model(linearSVM, dataloader, "Initialisation")

lr = 0.01
epochs = 1000
optimizer = optim.SGD(linearSVM.parameters(), lr=lr)
loss_fn = ExponentialLoss()

for _ in range(epochs):
    for x, y in dataloader:
        pred_y = linearSVM(x)
        optimizer.zero_grad()
        loss = loss_fn(pred_y[:, 0], y)
        loss.backward()
        optimizer.step()

    if (_+1) in [10, 100, 1000]:
        visualize_model(linearSVM, dataloader, title=f"Epoch: {_+1}; Loss: {loss.item()}")

