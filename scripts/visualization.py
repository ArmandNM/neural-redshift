import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn

from inductive_biases.models.neural_networks import NeuralNetwork, initialize_weights
from inductive_biases.models.activations import SinActivation, GaussianActivation
from inductive_biases.estimations import OutputSpaceEstimator


MIN_AMPLITUDE = 0.01
MAX_AMPLITUDE = 100
AMPLITUDE_SPACING = "log"  # "linear" or "log"
STEPS_AMPLITUDE = 15
MAX_HIDDEN_LAYERS = 7
EMBEDDING_DIM = 64
X1, X2, Y1, Y2 = -5, 5, -5, 5
POINTS_PER_AXIS = 300
BATCH_SIZE = 16384
SUPPORTED_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "swish": nn.SiLU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
    "sine": SinActivation,
    "gaussian": GaussianActivation,
}


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize_output_space(activation: str = "relu"):
    assert AMPLITUDE_SPACING in ["linear", "log"]
    if AMPLITUDE_SPACING == "linear":
        amplitudes = np.linspace(MIN_AMPLITUDE, MAX_AMPLITUDE, STEPS_AMPLITUDE)
    elif AMPLITUDE_SPACING == "log":
        amplitudes = np.logspace(np.log10(MIN_AMPLITUDE), np.log10(MAX_AMPLITUDE), STEPS_AMPLITUDE)
    else:
        raise ValueError(f"AMPLITUDE_SPACING must be 'linear' or 'log'.")

    fig, axs = plt.subplots(
        STEPS_AMPLITUDE,
        MAX_HIDDEN_LAYERS,
        figsize=(MAX_HIDDEN_LAYERS * 4, STEPS_AMPLITUDE * 4),
        constrained_layout=True,
    )

    estimator = OutputSpaceEstimator(x1=X1, y1=Y1, x2=X2, y2=Y2, steps=POINTS_PER_AXIS)

    for i, amplitude in enumerate(amplitudes):
        for j in range(MAX_HIDDEN_LAYERS):
            print(f"Amplitude: {amplitude:.2f}, Hidden Layers: {j + 1}")
            # Create model
            model = NeuralNetwork(
                num_layers=j + 3,
                dimensions=[2] + [EMBEDDING_DIM] * (j + 1) + [1],
                activation=SUPPORTED_ACTIVATIONS[activation],
            ).to(DEVICE)

            # Initialize random weights
            initialize_weights(
                model, weight_mode="uniform", bias_mode="uniform", amplitude=amplitude
            )

            # Estimate output space
            output_space = estimator.estimate(model, batch_size=BATCH_SIZE)

            # Plot output space
            axs[i, j].imshow(
                output_space,
                cmap="gray",
                interpolation="none",
                origin="lower",
                aspect="equal",
                extent=[X1, X2, Y1, Y2],
            )
            axs[i, j].axis("off")

    plt.savefig(f"nn{EMBEDDING_DIM}_{activation}.png")


def main():
    for activation in SUPPORTED_ACTIVATIONS.keys():
        visualize_output_space(activation=activation)


if __name__ == "__main__":
    main()
