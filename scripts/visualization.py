import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import torch
from torch import nn

from inductive_biases.models.neural_networks import NeuralNetwork, initialize_weights
from inductive_biases.models.activations import SinActivation, GaussianActivation
from inductive_biases.estimations import LinearDecisionSpaceEstimator
from inductive_biases.spectral_analysis import get_mean_frequency_2d

MIN_AMPLITUDE = 0.10
MAX_AMPLITUDE = 100
AMPLITUDE_SPACING = "log"  # "linear" or "log"
STEPS_AMPLITUDE = 15
MAX_HIDDEN_LAYERS = 7
EMBEDDING_DIM = 64
X1, X2, Y1, Y2 = -1, 1, -1, 1
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


def draw_outputs(spaces: dict, x_labels: list, y_labels: list, name: str):
    aspect_ratio = spaces[(0, 0)].shape[0] / spaces[(0, 0)].shape[1]

    fig, axs = plt.subplots(
        STEPS_AMPLITUDE,
        MAX_HIDDEN_LAYERS,
        figsize=(MAX_HIDDEN_LAYERS * 4 / aspect_ratio, STEPS_AMPLITUDE * 4),
        constrained_layout=True,
    )

    for i in range(STEPS_AMPLITUDE):
        for j in range(MAX_HIDDEN_LAYERS):
            axs[i, j].imshow(
                spaces[(i, j)],
                cmap="gray",
                interpolation="none",
                origin="lower",
                aspect="equal",
                extent=[X1, X2, Y1, Y2],
            )

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            if i == STEPS_AMPLITUDE - 1:
                axs[i, j].set_xlabel(x_labels[j])
            if j == 0:
                axs[i, j].set_ylabel(y_labels[i])

    plt.savefig(name, dpi=300)


def draw_mean_freqs(mean_freqs: torch.Tensor, x_labels: list, y_labels: list, name: str):
    fig, ax = plt.subplots(1, 1)

    img = ax.imshow(
        mean_freqs,
        cmap="Spectral",
        interpolation="none",
        origin="upper",
        aspect="equal",
        vmin=0,
        vmax=0.5,
    )

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Number of Hidden Layers")
    ax.set_ylabel("Maximum Weights Amplitude")
    ax.set_title(name.split(".")[0])
    fig.colorbar(img)
    plt.savefig(name, bbox_inches="tight", dpi=300)


def draw_spectra(spectral_spaces: dict, x_labels: list, y_labels: list, name: str):
    aspect_ratio = spectral_spaces[(0, 0)].shape[0] / spectral_spaces[(0, 0)].shape[1]

    fig, axs = plt.subplots(
        STEPS_AMPLITUDE,
        MAX_HIDDEN_LAYERS,
        figsize=(MAX_HIDDEN_LAYERS * 4 / aspect_ratio, STEPS_AMPLITUDE * 4),
        constrained_layout=True,
    )

    for i in range(STEPS_AMPLITUDE):
        for j in range(MAX_HIDDEN_LAYERS):
            axs[i, j].imshow(
                spectral_spaces[(i, j)],
                cmap="hot",
                interpolation="none",
                origin="upper",
                norm=matplotlib.colors.LogNorm(),
            )

            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            if i == STEPS_AMPLITUDE - 1:
                axs[i, j].set_xlabel(x_labels[j])
            if j == 0:
                axs[i, j].set_ylabel(y_labels[i])

    plt.savefig(name, dpi=300)


def analyze_output_space(activation: str = "relu"):
    assert AMPLITUDE_SPACING in ["linear", "log"]
    if AMPLITUDE_SPACING == "linear":
        amplitudes = np.linspace(MIN_AMPLITUDE, MAX_AMPLITUDE, STEPS_AMPLITUDE)
    elif AMPLITUDE_SPACING == "log":
        amplitudes = np.logspace(np.log10(MIN_AMPLITUDE), np.log10(MAX_AMPLITUDE), STEPS_AMPLITUDE)
    else:
        raise ValueError(f"AMPLITUDE_SPACING must be 'linear' or 'log'.")

    estimator = LinearDecisionSpaceEstimator(start=(X1, Y1), end=(X2, Y2), steps=POINTS_PER_AXIS)

    output_spaces = {}
    spectral_spaces = {}
    mean_freqs = torch.zeros((STEPS_AMPLITUDE, MAX_HIDDEN_LAYERS))

    # Estimate outputs spaces
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
                model,
                weight_mode="uniform",
                bias_mode="uniform",
                W_amplitude=amplitude,
                b_amplitude=amplitude,
            )

            # Estimate output space
            output_space = estimator.estimate(model, batch_size=BATCH_SIZE)
            output_spaces[(i, j)] = output_space.detach().cpu().numpy()

            # Compute Fourier decomposition
            coeff_2d = torch.fft.rfft2(output_space)
            coeff_2d = abs(coeff_2d.detach()).cpu()
            # Ignore 0 "Hz" frequency, as shift is not important
            coeff_2d[0, 0] = 0
            # Ignore lower half
            coeff_2d = coeff_2d[: coeff_2d.shape[0] // 2 + 1]
            spectral_spaces[(i, j)] = coeff_2d
            mean_freqs[i, j] = get_mean_frequency_2d(coeff_2d)

    x_labels = list(range(1, MAX_HIDDEN_LAYERS + 1))
    y_labels = [f"{amp:.2f}" for amp in amplitudes]
    draw_outputs(
        output_spaces,
        name=f"outputs_nn{EMBEDDING_DIM}_{activation}.png",
        x_labels=x_labels,
        y_labels=y_labels,
    )
    draw_mean_freqs(
        mean_freqs,
        name=f"mean_freqs_nn{EMBEDDING_DIM}_{activation}.png",
        x_labels=x_labels,
        y_labels=y_labels,
    )
    draw_spectra(
        spectral_spaces,
        name=f"spectra_nn{EMBEDDING_DIM}_{activation}.png",
        x_labels=x_labels,
        y_labels=y_labels,
    )


def main():
    for activation in SUPPORTED_ACTIVATIONS.keys():
        analyze_output_space(activation=activation)


if __name__ == "__main__":
    main()
