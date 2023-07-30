from torch import nn

from typing import List


SUPPORTED_ACTIVATIONS = [nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh]


class NeuralNetwork(nn.Module):
    def __init__(self, num_layers: int, dimensions: List[int], activation: nn.Module):
        super().__init__()

        assert num_layers == len(dimensions), (
            f"Number of layers ({num_layers}) must match ",
            f"number of dimensions given ({len(dimensions)}).",
        )

        assert activation in SUPPORTED_ACTIVATIONS, f"Activation {activation} not supported."

        layers = []
        for i in range(num_layers - 2):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))

        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_stack(x)
