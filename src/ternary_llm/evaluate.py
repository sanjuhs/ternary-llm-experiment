from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ternary_llm.config import VALID_MODES, ModelConfig, load_config
from ternary_llm.model import TernaryGPT
from ternary_llm.runtime import TokenStream, evaluate_model, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batches", type=int)
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        help="evaluate the same checkpoint weights under a different quantization mode",
    )
    parser.add_argument("--projection", type=Path)
    args = parser.parse_args()

    file_config = load_config(args.config)
    device = resolve_device(args.device or file_config.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stored = checkpoint["config"]
    model_config = ModelConfig(**stored["model"])
    evaluation_mode = args.mode or stored["mode"]
    model = TernaryGPT(model_config, evaluation_mode).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    if args.projection:
        model.load_activation_projection(str(args.projection))

    validation = TokenStream(
        file_config.data.validation_bin,
        model_config.context_length,
    )
    metrics = evaluate_model(
        model,
        validation,
        batch_size=file_config.train.batch_size,
        batches=args.batches or file_config.train.eval_batches,
        device=device,
        seed=file_config.seed + 10_000,
        precision=file_config.train.precision,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "step": checkpoint["step"],
        "processed_tokens": checkpoint["processed_tokens"],
        "checkpoint_mode": stored["mode"],
        "mode": evaluation_mode,
        "projection": str(args.projection) if args.projection else None,
        **metrics,
        "weight_codes": model.quantization_stats(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
