import pytest

import torch
from torch import nn

from inductive_biases.models.neural_networks import NeuralNetwork


def test_matching_number_of_layers_and_dimensions():
    with pytest.raises(AssertionError):
        NeuralNetwork(num_layers=2, dimensions=[1, 2, 3], activation=nn.ReLU)


def test_supported_activations():
    with pytest.raises(AssertionError):
        NeuralNetwork(num_layers=2, dimensions=[1, 2], activation=nn.Softmax)


def test_forward():
    model = NeuralNetwork(num_layers=3, dimensions=[1, 10, 1], activation=nn.ReLU)
    assert model.forward(torch.Tensor(1))


def test_batch_forward():
    model = NeuralNetwork(num_layers=3, dimensions=[1, 10, 1], activation=nn.ReLU)
    try:
        model.forward(torch.Tensor([[1], [2], [3]]))
    except:
        pytest.fail("Batch forward failed")
