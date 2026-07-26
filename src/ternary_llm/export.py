from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ternary_llm.packed import pack_ternary_codes, unpack_ternary_codes
from ternary_llm.quantization import (
    ternary_code,
    uses_quantized_activations,
    uses_ternary_weights,
)


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


def _pack_positive_scale(log_scale: Tensor) -> dict[str, Any]:
    positive = log_scale.detach().to(torch.float32).cpu().exp()
    unit = (positive.max() / 32767.0).clamp_min(torch.finfo(torch.float32).eps)
    codes = (positive / unit).round().clamp(1, 32767).to(torch.int16)
    return {
        "kind": "positive-scale-int16-v1",
        "codes": codes,
        "unit": unit,
        "shape": list(log_scale.shape),
        "source_parameterization": "natural_log",
    }


def pack_deployment_state_dict(
    state_dict: dict[str, Tensor],
    *,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Pack a model state without pretending auxiliary scales are ternary.

    Matrix, embedding, normalization, and fixed-Hadamard tensors follow the
    model's scaled ternary forward representation. Learned Q/K/V head scales
    are positive auxiliary values, so they use an explicit INT16 fixed-point
    representation. Non-floating buffers are preserved losslessly.
    """
    packed_state: dict[str, dict[str, Any]] = {}
    for name, tensor in state_dict.items():
        source = tensor.detach().cpu()
        if name.endswith(".attention.qkv_log_scales"):
            packed_state[name] = _pack_positive_scale(source)
            continue
        if not source.is_floating_point():
            packed_state[name] = {
                "kind": "raw-buffer-v1",
                "value": source,
                "shape": list(source.shape),
                "dtype": str(source.dtype),
            }
            continue
        scale = _scale(source.to(torch.float32))
        codes = ternary_code(source.to(torch.float32) / scale, threshold).to(torch.int8)
        original_shape = list(codes.shape)
        flattened = (
            codes.reshape(-1, codes.shape[-1])
            if codes.ndim > 1
            else codes.reshape(1, -1)
        )
        packed, width = pack_ternary_codes(flattened)
        packed_state[name] = {
            "kind": "scaled-ternary-2bit-v1",
            "packed_codes": packed,
            "scale": scale.to(torch.float16),
            "shape": original_shape,
            "packed_width": width,
        }
    return packed_state


def unpack_deployment_state_dict(
    packed_state: dict[str, dict[str, Any]],
) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, item in packed_state.items():
        kind = item["kind"]
        if kind == "scaled-ternary-2bit-v1":
            codes = unpack_ternary_codes(
                item["packed_codes"],
                int(item["packed_width"]),
            )
            codes = codes.reshape(item["shape"]).to(torch.float32)
            state[name] = codes * item["scale"].to(torch.float32)
        elif kind == "positive-scale-int16-v1":
            positive = item["codes"].to(torch.float32) * item["unit"].to(
                torch.float32
            )
            state[name] = positive.clamp_min(torch.finfo(torch.float32).tiny).log()
        elif kind == "raw-buffer-v1":
            state[name] = item["value"]
        else:
            raise ValueError(f"unsupported deployment tensor encoding: {kind}")
    return state


def inference_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Describe which strict inference claims are established by a checkpoint."""
    mode = config["mode"]
    model = config["model"]
    checks = {
        "ternary_forward_weights": uses_ternary_weights(mode),
        "quantized_residual_boundaries": uses_quantized_activations(mode),
        "residuals_use_binary_or_ternary_planes": model.get("activation_encoding")
        in {"residual_binary", "residual_ternary"},
        "ternary_qkv": model.get("qkv_quantization")
        in {"ternary", "binary_qk_ternary_v"},
        "factorizable_shared_qkv_scales": model.get("qkv_scale_granularity")
        == "learned_head",
        "integer_lut_two_bit_attention": model.get("attention_quantization")
        == "score_lut_prob_int2",
        "fixed_hadamard_projection": str(mode).startswith("hadamard_"),
        "no_floating_attention_rectification": model.get("attention_rectification")
        == "none",
        "no_sigmoid_attention_gate": model.get("attention_gate") == "none",
        "integer_friendly_ffn_activation": model.get("feed_forward_activation")
        == "relu",
        "integer_rms_norm": model.get("rms_norm_quantization")
        == "integer_reference",
        "dropout_disabled": float(model.get("dropout", 0.0)) == 0.0,
    }
    violations = [name for name, passed in checks.items() if not passed]
    remaining_boundaries = [
        "requantization scale arithmetic in the PyTorch quality path",
        "final token-sampling softmax",
    ]
    if not checks["integer_rms_norm"]:
        remaining_boundaries.insert(
            0,
            "integer RMSNorm reference is not wired into this checkpoint runtime",
        )
    return {
        "ternary_operand_contract": {
            "satisfied": not violations,
            "checks": checks,
            "violations": violations,
        },
        "end_to_end_integer_reference": {
            "satisfied": False,
            "remaining_boundaries": remaining_boundaries,
            "note": (
                "The artifact proves packed operands and declares auxiliary "
                "fixed-point scales; it is not a fused ASIC runtime."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    threshold = float(config["model"]["weight_threshold"])
    packed_state = pack_deployment_state_dict(
        checkpoint["model"],
        threshold=threshold,
    )
    encoding_counts: dict[str, int] = {}
    for item in packed_state.values():
        kind = str(item["kind"])
        encoding_counts[kind] = encoding_counts.get(kind, 0) + 1
    artifact = {
        "format": "ternary-deployment-v2",
        "source_checkpoint": str(args.checkpoint),
        "step": checkpoint["step"],
        "processed_tokens": checkpoint["processed_tokens"],
        "config": config,
        "inference_contract": inference_contract(config),
        "encoding_counts": encoding_counts,
        "model": packed_state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    result = {
        "output": str(args.output),
        "source_bytes": args.checkpoint.stat().st_size,
        "packed_bytes": args.output.stat().st_size,
        "tensor_count": len(packed_state),
        "encoding_counts": encoding_counts,
        "inference_contract": artifact["inference_contract"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
