import torch
import torch.nn as nn
import torch.optim as optim

class LeNetImpl(nn.Module):
  """
  LeNet-5 implementation on MNIST (28, 28)
  """
  def __init__(self, n_classes: int):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
        nn.ReLU(), # Original LeNet used tanh but will use ReLU
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16*5*5, 120), 
        nn.ReLU(),
        nn.Linear(120, 84), 
        nn.ReLU(),
        nn.Linear(84, n_classes)
    )

  def forward(self, x):
    return self.net(x)

# TODO: impl training and test loops
model = LeNetImpl(10)
optimizer = optim.Adam(model.parameters(), lr=1.e-3)
l = nn.CrossEntropyLoss()
