from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ternary_llm.config import (
    VALID_RMS_NORM_QUANTIZATIONS,
    ModelConfig,
    load_config,
)
from ternary_llm.model import (
    CausalSelfAttention,
    FeedForward,
    TernaryGPT,
    TernaryRMSNorm,
    TransformerBlock,
)
from ternary_llm.runtime import TokenStream, resolve_device


def tensor_stats(tensor: Tensor) -> dict[str, Any]:
    values = tensor.detach().to(torch.float32)
    return {
        "shape": list(values.shape),
        "zero_fraction": float((values == 0).to(torch.float32).mean().item()),
        "mean_abs": float(values.abs().mean().item()),
        "rms": float(values.square().mean().sqrt().item()),
        "feature_std": float(values.std(dim=-1).mean().item()),
    }


@torch.no_grad()
def activation_diagnostics(
    model: TernaryGPT,
    token_ids: Tensor,
) -> dict[str, Any]:
    boundaries: dict[str, dict[str, Any]] = {}
    block_outputs: list[Tensor] = []
    handles: list[Any] = []

    def capture(name: str, module: nn.Module):
        def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            boundaries[name] = {"module": type(module).__name__, **tensor_stats(output)}
            if isinstance(module, TransformerBlock):
                block_outputs.append(output.detach())

        return hook

    observed = (TernaryRMSNorm, CausalSelfAttention, FeedForward, TransformerBlock)
    for name, module in model.named_modules():
        if isinstance(module, observed):
            handles.append(module.register_forward_hook(capture(name, module)))
    try:
        logits, _ = model(token_ids)
    finally:
        for handle in handles:
            handle.remove()

    code_change = []
    for index in range(1, len(block_outputs)):
        previous = block_outputs[index - 1].sign()
        current = block_outputs[index].sign()
        code_change.append(
            {
                "from_block": index - 1,
                "to_block": index,
                "sign_change_fraction": float(
                    (previous != current).to(torch.float32).mean().item()
                ),
            }
        )
    probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return {
        "boundaries": boundaries,
        "residual_sign_change": code_change,
        "mean_output_entropy": float(entropy.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--rms-norm-quantization",
        choices=VALID_RMS_NORM_QUANTIZATIONS,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stored = checkpoint["config"]
    model_config = ModelConfig(**stored["model"])
    if args.rms_norm_quantization:
        model_config = replace(
            model_config,
            rms_norm_quantization=args.rms_norm_quantization,
        )
    model = TernaryGPT(model_config, stored["mode"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stream = TokenStream(config.data.validation_bin, model.config.context_length)
    generator = torch.Generator().manual_seed(config.seed + 20_000)
    token_ids, _ = stream.batch(args.batch_size, device=device, generator=generator)
    result = {
        "checkpoint": str(args.checkpoint),
        "mode": stored["mode"],
        "step": checkpoint["step"],
        **activation_diagnostics(model, token_ids),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
