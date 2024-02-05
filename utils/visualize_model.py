import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_model2D(models, data_loader, title):
  """
    - models: list of models
    - data_loder: DataLoader returning batches of pairs (x,y) with x being two-dimensional and y being +1/-1
  """
  num_models = len(models)
  if(num_models==1):
    model = models[0]
    model.to("cpu")
    model.eval()
    data, labels = next(iter(data_loader))

    data1 = data[labels == 1]
    data0 = data[labels == -1]
    plt.scatter(data1[:, 0], data1[:, 1], c='r', marker='o', edgecolors='k', s=50)
    plt.scatter(data0[:, 0], data0[:, 1], c='b', marker='x', edgecolors='k', s=50)
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
    plt.title(title)
    plt.show()
    return 0

  fig, axes = plt.subplots(1, num_models, figsize=(12, 8))
  for i in range(num_models):
    model = models[i]
    model.eval()
    data, labels = next(iter(data_loader))

    data1 = data[labels == 1]
    data0 = data[labels == -1]
    axes[i].scatter(data1[:, 0], data1[:, 1], c='r', marker='o', edgecolors='k', s=50)
    axes[i].scatter(data0[:, 0], data0[:, 1], c='b', marker='x', edgecolors='k', s=50)

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.tensor(mesh_data, dtype=torch.float32)

    with torch.no_grad():
        model.eval()
        predictions = torch.sign(model(mesh_tensor)).cpu().numpy()

    axes[i].contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.3, cmap='viridis')
  plt.title(title)
  plt.show()
