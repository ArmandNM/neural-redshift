import torch
from torch import nn


from inductive_biases.models.neural_networks import NeuralNetwork


def main():
    model = NeuralNetwork(num_layers=3, dimensions=[1, 10, 1], activation=nn.ReLU)
    x = torch.randn(5, 1)
    print(model(x))


if __name__ == "__main__":
    main()
