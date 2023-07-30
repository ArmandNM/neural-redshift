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


def initialize_weights(
    model: nn.Module, weight_mode: str = "uniform", bias_mode="uniform", amplitude=1.0
):
    assert weight_mode in ["uniform", "normal", "xavier_uniform", "xavier_normal"]
    assert bias_mode in ["uniform", "normal", "zero"]

    if weight_mode == "uniform":
        init_fn = lambda weight: weight.data.uniform_(-amplitude, amplitude)
    elif weight_mode == "xavier_uniform":
        init_fn = lambda tensor: nn.init.xavier_uniform_(tensor, gain=amplitude)
    elif weight_mode == "normal":
        init_fn = lambda weight: weight.data.normal_(mean=0, std=amplitude)
    elif weight_mode == "xavier_normal":
        init_fn = lambda tensor: nn.init.xavier_normal_(tensor, gain=amplitude)
    else:
        raise ValueError(f"Invalid distribution {weight_mode}")

    if bias_mode == "uniform":
        bias_init = lambda bias: bias.data.uniform_(-amplitude, amplitude)
    elif bias_mode == "normal":
        bias_init = lambda bias: bias.data.normal_(mean=0, std=amplitude)
    elif bias_mode == "zero":
        bias_init = lambda bias: bias.data.zero_()
    else:
        raise ValueError(f"Invalid bias_mode {bias_mode}")

    # Initialize weights and biases for all linear layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_
            init_fn(module.weight)
            if module.bias is not None:
                bias_init(module.bias)
