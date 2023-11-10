import argparse
import torch
import math
import json
import random
import numpy as np
import os

from torch import nn

from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from tqdm import tqdm

from inductive_biases.models.neural_networks import initialize_weights
from inductive_biases.models.activations import SinActivation, GaussianActivation
from inductive_biases.complexity import lempel_ziv_complexity


device = "cuda" if torch.cuda.is_available() else "cpu"

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


def replace_layers(model, args):
    def _replace_layer(module):
        if isinstance(module, torch.nn.LayerNorm):
            if args.disable_layernorm:
                return torch.nn.Identity()
            elif args.modulate_layernorm:
                module.weight.data = torch.ones_like(module.weight.data) * args.layernorm_gamma
                module.bias.data = torch.zeros_like(module.bias.data)
                return module
            return module
        elif isinstance(module, torch.nn.ReLU):
            return SUPPORTED_ACTIVATIONS[args.activation]()
        else:
            return module

    def _recursive_setattr(obj, attr, value):
        attr = attr.split(".", 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            _recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    for name, module in tuple(model.named_modules()):
        if name:
            _recursive_setattr(model, name, _replace_layer(module))


def prune_layers(model, num_layers):
    model.transformer.h = nn.ModuleList(model.transformer.h[:num_layers])


def replace_positional_encodings(model):
    max_len, embed_dim = model.transformer.wpe.weight.shape
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
    model.transformer.wpe.weight.data *= 0  # torch.zeros_like(model.transformer.wpe.weight)
    model.transformer.wpe.weight.data[:, 0::2] = torch.sin(position * div_term)
    model.transformer.wpe.weight.data[:, 1::2] = torch.cos(position * div_term)
    # Scale down positional embeddings as they are much larger than token embeddings
    model.transformer.wpe.weight.data /= 33.33


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to suppress warning
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.use_deterministic_algorithms(True)


def write_results(args, complexities):
    exp_name = ""
    exp_name += f"{args.model_name}__"
    exp_name += f"{args.activation}__"
    exp_name += f"gamma_{args.layernorm_gamma}__"
    exp_name += f"layers_{args.num_layers}__"
    if args.use_positional_encodings:
        exp_name += "trig_enc__"
    exp_name += f"{args.seed}"

    results = {}
    results.update(vars(args))
    results["mean"] = complexities.mean().item()
    results["std"] = complexities.std().item()
    results["complexities"] = complexities.tolist()

    with open(f"{args.output_path}/{exp_name}.json", "w") as f:
        json.dump(results, f, indent=4)


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
    parser.add_argument("--output_path", type=str, required=False, default="./results")

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--disable_layernorm", action="store_true")
    parser.add_argument("--modulate_layernorm", action="store_true")
    parser.add_argument("--use_positional_encodings", action="store_true")

    args = parser.parse_args()

    # args.modulate_layernorm = True
    # args.use_positional_encodings = False

    set_random_seed(args.seed)

    # Instantiate tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

    # Instantiate model for causal language modeling
    model = GPT2LMHeadModel.from_pretrained(
        args.model_name,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
        activation_function="relu",
    )

    # Initialize weights
    model.apply(model._init_weights)
    # initialize_weights(model.transformer.h, W_amplitude=args.amplitude, b_amplitude=args.amplitude)
    # initialize_weights(model.lm_head, weight_mode="normal", W_amplitude=0.02, bias_mode="zero")

    # Use trigonometric encodings instead of learned positional embeddings
    if args.use_positional_encodings:
        replace_positional_encodings(model)

    # Fix number of layers
    prune_layers(model, args.num_layers)

    # Replace or modulate layers
    replace_layers(model.transformer.h, args)

    model.to(device)
    model.eval()

    prompts = []
    prompts = torch.tensor([[i] for i in range(len(tokenizer.vocab))]).long()

    dataloader = torch.utils.data.DataLoader(prompts[:1000], batch_size=args.batch_size)
    outputs = []

    generated_ids = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            outputs = model.generate(
                data.to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                top_k=args.top_k,
            )
            generated_ids.append(outputs.to("cpu"))
            del outputs
            torch.cuda.empty_cache()

    generated_ids = torch.cat(generated_ids, dim=0)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    complexities = []

    for gids in generated_ids:
        gids = gids.tolist()
        complexities.append(lempel_ziv_complexity(gids) / len(gids))

    complexities = torch.tensor(complexities).float()
    write_results(args, complexities)


if __name__ == "__main__":
    main()
