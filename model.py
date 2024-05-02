import torch
import torch.nn as nn
import torch.optim as optim

class PINO(nn.Module):
    def __init__(self):
        super(PINO, self).__init__()
        self.linear = nn.Linear(1, 1)  # Input size: 1, Output size: 1

    def forward(self, x):
        return self.linear(x)