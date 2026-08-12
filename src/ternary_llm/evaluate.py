from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from ternary_llm.config import (
    VALID_ACTIVATION_ENCODINGS,
    VALID_ATTENTION_GATES,
    VALID_ATTENTION_NORMALIZATIONS,
    VALID_ATTENTION_QUANTIZATIONS,
    VALID_ATTENTION_RECTIFICATIONS,
    VALID_FEED_FORWARD_ACTIVATIONS,
    VALID_MODES,
    VALID_QKV_QUANTIZATIONS,
    VALID_QKV_SCALE_GRANULARITIES,
    VALID_RMS_NORM_QUANTIZATIONS,
    ModelConfig,
    load_config,
)
from ternary_llm.model import TernaryGPT
from ternary_llm.runtime import (
    TokenStream,
    evaluate_model,
    evaluate_model_sequential,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batches", type=int)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="scan deterministic non-overlapping windows instead of random batches",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        help="evaluate the same checkpoint weights under a different quantization mode",
    )
    parser.add_argument("--projection", type=Path)
    parser.add_argument(
        "--attention-quantization",
        choices=VALID_ATTENTION_QUANTIZATIONS,
    )
    parser.add_argument(
        "--attention-normalization",
        choices=VALID_ATTENTION_NORMALIZATIONS,
    )
    parser.add_argument("--attention-clip", type=float)
    parser.add_argument("--attention-threshold", type=float)
    parser.add_argument("--qkv-quantization", choices=VALID_QKV_QUANTIZATIONS)
    parser.add_argument(
        "--qkv-scale-granularity",
        choices=VALID_QKV_SCALE_GRANULARITIES,
    )
    parser.add_argument("--qkv-scale-initial", type=float)
    parser.add_argument(
        "--attention-rectification",
        choices=VALID_ATTENTION_RECTIFICATIONS,
    )
    parser.add_argument("--attention-gate", choices=VALID_ATTENTION_GATES)
    parser.add_argument("--attention-gate-initial", type=float)
    parser.add_argument("--activation-levels", type=int)
    parser.add_argument("--activation-encoding", choices=VALID_ACTIVATION_ENCODINGS)
    parser.add_argument("--activation-planes", type=int)
    parser.add_argument(
        "--feed-forward-activation",
        choices=VALID_FEED_FORWARD_ACTIVATIONS,
    )
    parser.add_argument(
        "--rms-norm-quantization",
        choices=VALID_RMS_NORM_QUANTIZATIONS,
    )
    args = parser.parse_args()

    file_config = load_config(args.config)
    device = resolve_device(args.device or file_config.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stored = checkpoint["config"]
    model_config = ModelConfig(**stored["model"])
    model_overrides = {}
    if args.attention_quantization:
        model_overrides["attention_quantization"] = args.attention_quantization
    if args.attention_normalization:
        model_overrides["attention_normalization"] = args.attention_normalization
    if args.attention_clip is not None:
        model_overrides["attention_clip"] = args.attention_clip
    if args.attention_threshold is not None:
        model_overrides["attention_threshold"] = args.attention_threshold
    if args.qkv_quantization:
        model_overrides["qkv_quantization"] = args.qkv_quantization
    if args.qkv_scale_granularity:
        model_overrides["qkv_scale_granularity"] = args.qkv_scale_granularity
    if args.qkv_scale_initial is not None:
        model_overrides["qkv_scale_initial"] = args.qkv_scale_initial
    if args.attention_rectification:
        model_overrides["attention_rectification"] = args.attention_rectification
    if args.attention_gate:
        model_overrides["attention_gate"] = args.attention_gate
    if args.attention_gate_initial is not None:
        model_overrides["attention_gate_initial"] = args.attention_gate_initial
    if args.activation_levels is not None:
        model_overrides["activation_levels"] = args.activation_levels
    if args.activation_encoding:
        model_overrides["activation_encoding"] = args.activation_encoding
    if args.activation_planes is not None:
        model_overrides["activation_planes"] = args.activation_planes
    if args.feed_forward_activation:
        model_overrides["feed_forward_activation"] = args.feed_forward_activation
    if args.rms_norm_quantization:
        model_overrides["rms_norm_quantization"] = args.rms_norm_quantization
    if model_overrides:
        model_config = replace(model_config, **model_overrides)
    evaluation_mode = args.mode or stored["mode"]
    if args.projection and evaluation_mode.startswith("hadamard_"):
        raise SystemExit("--projection cannot be combined with a fixed Hadamard mode")
    model = TernaryGPT(model_config, evaluation_mode).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    if evaluation_mode.startswith("hadamard_"):
        model.reset_hadamard_projection()
    if args.projection:
        model.load_activation_projection(str(args.projection))

    validation = TokenStream(
        file_config.data.validation_bin,
        model_config.context_length,
    )
    if args.sequential:
        metrics = evaluate_model_sequential(
            model,
            validation,
            batch_size=file_config.train.batch_size,
            max_batches=args.batches,
            device=device,
            precision=file_config.train.precision,
        )
    else:
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
        "attention": model.attention_stats(),
        "residual": model.residual_stats(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
