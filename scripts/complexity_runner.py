import argparse
import numpy as np

from torch import nn

from inductive_biases.models.activations import SinActivation, GaussianActivation
from transformers_complexity import compute_complexity


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

# GAMMA = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10, 50, 100, 1000]
GAMMAS = np.logspace(-1, 3, 25).tolist()
LAYERS = [1, 3, 6, 9, 12]
SEEDS = [1, 2, 3, 4, 5]

DISABLE_LAYERNORM = False
MODULATE_LAYERNORM = True
USE_POSITIONAL_ENCODINGS = False

MODEL_NAME = "gpt2"

def main():
    parser = argparse.ArgumentParser(description="Transformers complexity analysis")
    parser.add_argument("--model_name", type=str, required=False, default="gpt2")
    parser.add_argument("--activation", type=str, required=False, default="tanh")
    parser.add_argument("--amplitude", type=float, required=False, default=0.02)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--top_k", type=int, required=False, default=50)
    parser.add_argument("--layernorm_gamma", type=float, default=1.0)
    parser.add_argument("--num_layers", type=int, required=False, default=12)
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--max_inferences", type=int, default=1000)
    parser.add_argument("--output_path", type=str, required=False, default="./results")

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--disable_layernorm", action="store_true")
    parser.add_argument("--modulate_layernorm", action="store_true")
    parser.add_argument("--use_positional_encodings", action="store_true")

    args = parser.parse_args()

    args.disable_layernorm = DISABLE_LAYERNORM
    args.modulate_layernorm = MODULATE_LAYERNORM
    args.use_positional_encodings = USE_POSITIONAL_ENCODINGS

    SELECTED_ACTIVATIONS = [args.activation]

    num_experiments = len(LAYERS) * len(GAMMAS) * len(SEEDS) * len(SELECTED_ACTIVATIONS)
    print(f"Found {num_experiments} to run.")

    for activation in SELECTED_ACTIVATIONS:
        for num_layers in LAYERS:
            for gamma in GAMMAS:
                for seed in SEEDS:
                    print(f"Running activation: {activation} num_layers: {num_layers} gamma: {gamma} seed: {seed}")

                    args.activation = activation
                    args.num_layers = num_layers
                    args.layernorm_gamma = gamma
                    args.seed = seed

                    compute_complexity(args)

    print("Done")


if __name__ == "__main__":
    main()
