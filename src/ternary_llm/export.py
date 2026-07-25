from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ternary_llm.packed import pack_ternary_codes, unpack_ternary_codes
from ternary_llm.quantization import ternary_code


def _scale(tensor: Tensor) -> Tensor:
    dimensions: int | tuple[int, ...]
    dimensions = 0 if tensor.ndim <= 1 else tuple(range(1, tensor.ndim))
    return tensor.abs().mean(dim=dimensions, keepdim=True).clamp_min(1e-5)


def pack_state_dict(
    state_dict: dict[str, Tensor],
    *,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    packed_state: dict[str, dict[str, Any]] = {}
    for name, tensor in state_dict.items():
        source = tensor.detach().to(torch.float32).cpu()
        scale = _scale(source)
        codes = ternary_code(source / scale, threshold).to(torch.int8)
        original_shape = list(codes.shape)
        flattened = codes.reshape(-1, codes.shape[-1]) if codes.ndim > 1 else codes.reshape(1, -1)
        packed, width = pack_ternary_codes(flattened)
        packed_state[name] = {
            "packed_codes": packed,
            "scale": scale.to(torch.float16),
            "shape": original_shape,
            "packed_width": width,
        }
    return packed_state


def unpack_state_dict(packed_state: dict[str, dict[str, Any]]) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, item in packed_state.items():
        codes = unpack_ternary_codes(item["packed_codes"], int(item["packed_width"]))
        codes = codes.reshape(item["shape"]).to(torch.float32)
        state[name] = codes * item["scale"].to(torch.float32)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    threshold = float(config["model"]["weight_threshold"])
    packed_state = pack_state_dict(checkpoint["model"], threshold=threshold)
    artifact = {
        "format": "ternary-2bit-v1",
        "source_checkpoint": str(args.checkpoint),
        "step": checkpoint["step"],
        "processed_tokens": checkpoint["processed_tokens"],
        "config": config,
        "model": packed_state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    result = {
        "output": str(args.output),
        "source_bytes": args.checkpoint.stat().st_size,
        "packed_bytes": args.output.stat().st_size,
        "tensor_count": len(packed_state),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
