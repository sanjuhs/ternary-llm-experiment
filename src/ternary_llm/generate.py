from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ternary_llm.config import ModelConfig
from ternary_llm.model import TernaryGPT
from ternary_llm.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stored = checkpoint["config"]
    model = TernaryGPT(ModelConfig(**stored["model"]), stored["mode"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if bos_id is not None:
        prompt_ids.insert(0, bos_id)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    torch.manual_seed(args.seed)
    generated = model.generate(
        tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=None if args.top_k <= 0 else args.top_k,
        eos_id=eos_id,
    )
    print(tokenizer.decode(generated[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
