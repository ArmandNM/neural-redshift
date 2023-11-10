import subprocess
import sys

from torch import nn

from inductive_biases.models.activations import SinActivation, GaussianActivation


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

GAMMA = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10, 50, 100, 1000]
AMPLITUDE = 0.02
BATCH_SIZE = 256
MAX_NEW_TOKENS = 50
DO_SAMPLE = False
TOP_K = 50
MIN_LAYERS = 6
MAX_LAYERS = 12
MAX_SEEDS = 3

DISABLE_LAYERNORM = False
MODULATE_LAYERNORM = True

MODEL_NAME = "gpt2"

def main():
    # Specify which activations to run as command line arguments
    if len(sys.argv) > 1:
        SELECTED_ACTIVATIONS = []
        for arg in sys.argv:
            if arg in SUPPORTED_ACTIVATIONS.keys():
                SELECTED_ACTIVATIONS.append(arg)
    else:
        # Run with all activations if not arguments are specified
        SELECTED_ACTIVATIONS = list(SUPPORTED_ACTIVATIONS.keys())

    print(f"Running with activations: {SELECTED_ACTIVATIONS}")

    # results = {}
    for activation in SELECTED_ACTIVATIONS:
        # results[activation] = {}
        for num_layers in range(MIN_LAYERS, MAX_LAYERS + 1):
            for gamma in GAMMA:
                # results[activation][gamma] = []
                for seed in range(MAX_SEEDS):
                    print(f"Running activation: {activation} num_layers: {num_layers} gamma: {gamma} seed: {seed}")
                    # outputs = subprocess.check_output(["python", "scripts/transformers_complexity.py"])
                    outputs = subprocess.check_output(["python", "scripts/transformers_complexity.py",
                                    "--model_name", MODEL_NAME,
                                    "--activation", activation,
                                    "--amplitude", f"{AMPLITUDE}",
                                    "--batch_size", f"{BATCH_SIZE}",
                                    "--max_new_tokens", f"{MAX_NEW_TOKENS}",
                                    "--top_k", f"{TOP_K}",
                                    "--layernorm_gamma", f"{gamma}",
                                    "--num_layers", f"{num_layers}",
                                    "--seed", f"{seed}",
                                    "--modulate_layernorm"
                                    ]
                                )
                    # print(outputs)
                    # mean, std = outputs.decode("UTF-8").split("\n")[0:2]
                    # mean = float(mean.split(": ")[1])
                    # std = float(std.split(": ")[1])
                    # results[activation][gamma].append((mean, std))

    # print(results)
    print("Done")


if __name__ == "__main__":
    main()
