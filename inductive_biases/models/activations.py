import torch
from torch import nn


class SinActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class GaussianActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.exp(-(x**2))
