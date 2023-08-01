import pytest

import torch
from torch import nn

from inductive_biases.models.neural_networks import NeuralNetwork, initialize_weights


class TestNeuralNetwork:
    def test_matching_number_of_layers_and_dimensions(self):
        with pytest.raises(AssertionError):
            NeuralNetwork(num_layers=2, dimensions=[1, 2, 3], activation=nn.ReLU)

    def test_supported_activations(self):
        with pytest.raises(AssertionError):
            NeuralNetwork(num_layers=2, dimensions=[1, 2], activation=nn.Softmax)

    def test_forward(self):
        model = NeuralNetwork(num_layers=3, dimensions=[1, 10, 1], activation=nn.ReLU)
        assert model.forward(torch.Tensor(1))

    def test_batch_forward(self):
        model = NeuralNetwork(num_layers=3, dimensions=[1, 10, 1], activation=nn.ReLU)
        try:
            model.forward(torch.Tensor([[1], [2], [3]]))
        except:
            pytest.fail("Batch forward failed")


class TestInitialization:
    def test_init_uniform_small_amplitude(self):
        model = NeuralNetwork(num_layers=3, dimensions=[1, 10, 1], activation=nn.ReLU)
        eps = 0.00001
        initialize_weights(model, weight_mode="uniform", bias_mode="uniform", amplitude=eps)
        assert model.linear_stack[0].weight.max() < eps
        assert model.linear_stack[0].bias.max() < eps
        assert model.linear_stack[0].weight.min() > -eps
        assert model.linear_stack[0].bias.min() > -eps

    def test_init_bias_zeros(self):
        model = NeuralNetwork(num_layers=3, dimensions=[1, 10, 1], activation=nn.ReLU)
        initialize_weights(model, bias_mode="zero")
        assert model.linear_stack[0].weight.detach().any()
        assert not model.linear_stack[0].bias.detach().any()

    def test_invalid_weight_mode(self):
        with pytest.raises(AssertionError):
            model = NeuralNetwork(num_layers=3, dimensions=[1, 10, 1], activation=nn.ReLU)
            initialize_weights(model, weight_mode="invalid_mode")

    def test_invalid_bias_mode(self):
        with pytest.raises(AssertionError):
            model = NeuralNetwork(num_layers=3, dimensions=[1, 10, 1], activation=nn.ReLU)
            initialize_weights(model, bias_mode="invalid_mode")
