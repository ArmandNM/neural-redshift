import argparse
import textwrap
import torch
from torch import nn

from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from lempel_ziv_complexity import lempel_ziv_complexity

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
        attr = attr.split('.', 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            _recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    for name, module in tuple(model.named_modules()):
        if name:
            _recursive_setattr(model, name, _replace_layer(module))


def main():
    parser = argparse.ArgumentParser(
        description="Transformers complexity analysis")
    parser.add_argument("--model_name", type=str, required=False, default="gpt2")
    parser.add_argument("--activation", type=str, required=False, default="relu")
    parser.add_argument("--amplitude", type=float, required=False, default=0.02)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--top_k", type=int, required=False, default=50)
    parser.add_argument("--layernorm_gamma", type=float, default=0.1)

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--disable_layernorm", action="store_true")
    parser.add_argument("--modulate_layernorm", action="store_true")

    args = parser.parse_args()

    # args.modulate_layernorm = True

    # Instantiate tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error

    # Instantiate model for causal language modeling
    model = GPT2LMHeadModel.from_pretrained(args.model_name,
                                            output_scores=True,
                                            pad_token_id=tokenizer.eos_token_id,
                                            activation_function="relu"
                                            )

    # Initialize weights
    model.apply(model._init_weights)
    # initialize_weights(model.transformer.h, W_amplitude=args.amplitude, b_amplitude=args.amplitude)
    # initialize_weights(model.lm_head, weight_mode="normal", W_amplitude=0.02, bias_mode="zero")

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
            outputs = model.generate(data.to(device),
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
    print(f"mean: {complexities.mean()}")
    print(f"std: {complexities.std()}")


if __name__ == "__main__":
    main()