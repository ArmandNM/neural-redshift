import torch
from torch import nn


from inductive_biases.models.neural_networks import NeuralNetwork, initialize_weights


def main():
    model = NeuralNetwork(num_layers=3, dimensions=[1, 10, 1], activation=nn.ReLU)
    initialize_weights(model, weight_mode="uniform", bias_mode="zero", amplitude=1)

    x = torch.randn(5, 1)
    print(model(x))


if __name__ == "__main__":
    main()
