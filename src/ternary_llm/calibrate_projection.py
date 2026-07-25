from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ternary_llm.config import ModelConfig, load_config
from ternary_llm.model import TernaryGPT, TransformerBlock
from ternary_llm.projection import (
    CovarianceAccumulator,
    coat_projection,
    covariance_diagonal_cv,
)
from ternary_llm.runtime import TokenStream, resolve_device


@torch.no_grad()
def calibrate(
    *,
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path,
    device_name: str,
    batches: int,
    batch_size: int,
) -> dict[str, Any]:
    config = load_config(config_path)
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stored = checkpoint["config"]
    model_config = ModelConfig(**stored["model"])
    model = TernaryGPT(model_config, stored["mode"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    accumulator = CovarianceAccumulator(model_config.d_model)
    handles = []

    def capture(_module: TransformerBlock, inputs: tuple[torch.Tensor, ...]) -> None:
        accumulator.update(inputs[0])

    for block in model.blocks:
        handles.append(block.register_forward_pre_hook(capture))

    stream = TokenStream(config.data.validation_bin, model_config.context_length)
    generator = torch.Generator().manual_seed(config.seed + 30_000)
    try:
        for _ in range(batches):
            token_ids, _ = stream.batch(
                batch_size,
                device=device,
                generator=generator,
            )
            model(token_ids)
    finally:
        for handle in handles:
            handle.remove()

    covariance = accumulator.covariance()
    projection = coat_projection(covariance)
    identity = torch.eye(model_config.d_model)
    orthogonality_error = float(
        (projection.T @ projection - identity).abs().max().item()
    )
    metadata = {
        "method": "COAT closed-form Q = U H",
        "source_checkpoint": str(checkpoint_path),
        "source_mode": stored["mode"],
        "source_step": int(checkpoint["step"]),
        "samples": accumulator.count,
        "d_model": model_config.d_model,
        "covariance_diagonal_cv_before": covariance_diagonal_cv(covariance),
        "covariance_diagonal_cv_after": covariance_diagonal_cv(covariance, projection),
        "max_orthogonality_error": orthogonality_error,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projection": projection,
            "mean": accumulator.mean.to(torch.float32),
            "covariance": covariance.to(torch.float32),
            "metadata": metadata,
        },
        output_path,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error("--batches and --batch-size must be positive")
    result = calibrate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output,
        device_name=args.device,
        batches=args.batches,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
