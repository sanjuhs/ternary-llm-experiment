from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

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
    ExperimentConfig,
    ModelConfig,
    load_config,
    replace_mode,
)
from ternary_llm.model import TernaryGPT
from ternary_llm.optimizers import (
    StochasticTernaryOptimizer,
    TernaryCounterOptimizer,
    optimizer_state_bytes,
)
from ternary_llm.runtime import (
    TokenStream,
    autocast_context,
    evaluate_model,
    learning_rate,
    resolve_device,
    seed_everything,
)


def save_checkpoint(
    path: Path,
    *,
    model: TernaryGPT,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    step: int,
    processed_tokens: int,
    batch_generator: torch.Generator,
    scaler: torch.amp.GradScaler,
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "processed_tokens": processed_tokens,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "batch_generator": batch_generator.get_state(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_checkpoint(
    path: Path,
    *,
    model: TernaryGPT,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_generator: torch.Generator,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if "batch_generator" in state["rng"]:
        batch_generator.set_state(state["rng"]["batch_generator"].cpu())
    return int(state["step"]), int(state["processed_tokens"])


def append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, sort_keys=True) + "\n")


def verify_tokenizer(config: ExperimentConfig) -> None:
    tokenizer_path = Path(config.data.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"tokenizer not found: {tokenizer_path}. Run `uv run ternary-data prepare`."
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    actual = tokenizer.get_vocab_size()
    if actual != config.model.vocab_size:
        raise ValueError(f"config vocab_size is {config.model.vocab_size}, tokenizer has {actual}")


def _distillation_enabled(config: ExperimentConfig) -> bool:
    return any(
        weight > 0
        for weight in (
            config.train.logit_distillation_weight,
            config.train.attention_distillation_weight,
            config.train.qk_distillation_weight,
            config.train.hidden_distillation_weight,
        )
    )


def _qk_similarity(tensor: torch.Tensor, stride: int) -> torch.Tensor:
    sampled = F.normalize(tensor[:, :, ::stride, :], dim=-1)
    return sampled @ sampled.transpose(-2, -1)


def distillation_loss(
    student: TernaryGPT,
    teacher: TernaryGPT,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine BWTA logit/attention KD with Q-ViT Q/K similarity KD."""
    temperature = config.train.distillation_temperature
    total = student_logits.new_zeros(())
    metrics: dict[str, float] = {}
    if config.train.logit_distillation_weight > 0:
        logit_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2 / student_logits.shape[1]
        total = total + config.train.logit_distillation_weight * logit_loss
        metrics["logit_distillation_loss"] = float(logit_loss.detach().item())

    attention_losses = []
    qk_losses = []
    hidden_losses = []
    for student_block, teacher_block in zip(
        student.blocks,
        teacher.blocks,
        strict=True,
    ):
        student_attention = student_block.attention
        teacher_attention = teacher_block.attention
        if config.train.attention_distillation_weight > 0:
            if (
                student_attention.last_probabilities is None
                or teacher_attention.last_probabilities is None
            ):
                raise AssertionError("attention probabilities were not captured")
            attention_losses.append(
                F.mse_loss(
                    student_attention.last_probabilities,
                    teacher_attention.last_probabilities,
                )
            )
        if config.train.qk_distillation_weight > 0:
            if any(
                tensor is None
                for tensor in (
                    student_attention.last_q,
                    student_attention.last_k,
                    teacher_attention.last_q,
                    teacher_attention.last_k,
                )
            ):
                raise AssertionError("Q/K tensors were not captured")
            stride = config.train.distillation_token_stride
            qk_losses.extend(
                (
                    F.mse_loss(
                        _qk_similarity(student_attention.last_q, stride),
                        _qk_similarity(teacher_attention.last_q, stride),
                    ),
                    F.mse_loss(
                        _qk_similarity(student_attention.last_k, stride),
                        _qk_similarity(teacher_attention.last_k, stride),
                    ),
                )
            )
        if config.train.hidden_distillation_weight > 0:
            if student_block.last_hidden is None or teacher_block.last_hidden is None:
                raise AssertionError("block outputs were not captured")
            hidden_shape = (student_block.last_hidden.shape[-1],)
            hidden_losses.append(
                F.mse_loss(
                    F.layer_norm(student_block.last_hidden, hidden_shape),
                    F.layer_norm(teacher_block.last_hidden, hidden_shape),
                )
            )

    if attention_losses:
        attention_loss = torch.stack(attention_losses).mean()
        total = total + config.train.attention_distillation_weight * attention_loss
        metrics["attention_distillation_loss"] = float(
            attention_loss.detach().item()
        )
    if qk_losses:
        qk_loss = torch.stack(qk_losses).mean()
        total = total + config.train.qk_distillation_weight * qk_loss
        metrics["qk_distillation_loss"] = float(qk_loss.detach().item())
    if hidden_losses:
        hidden_loss = torch.stack(hidden_losses).mean()
        total = total + config.train.hidden_distillation_weight * hidden_loss
        metrics["hidden_distillation_loss"] = float(hidden_loss.detach().item())
    return total, metrics


def train(
    config: ExperimentConfig,
    *,
    run_name: str | None = None,
    resume: Path | None = None,
    init_from: Path | None = None,
    projection: Path | None = None,
    teacher_checkpoint: Path | None = None,
) -> Path:
    seed_everything(config.seed)
    verify_tokenizer(config)
    device = resolve_device(config.device)
    output_dir = Path(config.train.output_dir)
    if run_name:
        output_dir = output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    resolved_path = output_dir / "resolved-config.json"
    resolved_path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    train_stream = TokenStream(config.data.train_bin, config.model.context_length)
    validation_stream = TokenStream(config.data.validation_bin, config.model.context_length)
    model = TernaryGPT(config.model, config.mode).to(device)
    if projection is not None and config.mode.startswith("hadamard_"):
        raise ValueError("a calibrated projection cannot be combined with a fixed Hadamard mode")
    if init_from is not None:
        initial = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(initial["model"], strict=False)
        if config.mode.startswith("hadamard_"):
            model.reset_hadamard_projection()
    if projection is not None:
        model.load_activation_projection(str(projection))
    teacher: TernaryGPT | None = None
    if _distillation_enabled(config):
        if teacher_checkpoint is None:
            raise ValueError(
                "distillation weights are non-zero but --teacher-checkpoint is missing"
            )
        teacher_state = torch.load(
            teacher_checkpoint,
            map_location=device,
            weights_only=False,
        )
        teacher_model_config = ModelConfig(**teacher_state["config"]["model"])
        teacher_model_config = replace(
            teacher_model_config,
            attention_quantization="float",
            qkv_quantization="inherit",
            attention_rectification="none",
            attention_gate="none",
        )
        teacher = TernaryGPT(
            teacher_model_config,
            teacher_state["config"]["mode"],
        ).to(device)
        teacher.load_state_dict(teacher_state["model"], strict=False)
        if teacher_state["config"]["mode"].startswith("hadamard_"):
            teacher.reset_hadamard_projection()
        if projection is not None:
            teacher.load_activation_projection(str(projection))
        teacher.eval()
        teacher.requires_grad_(False)
        teacher.set_capture_distillation(True)
        model.set_capture_distillation(True)
    if config.train.optimizer == "ternary_counter":
        optimizer: torch.optim.Optimizer = TernaryCounterOptimizer(
            model.parameters(),
            counter_threshold=config.train.counter_threshold,
        )
    elif config.train.optimizer == "stochastic_ternary":
        optimizer = StochasticTernaryOptimizer(
            model.parameters(),
            transition_rate=config.train.transition_rate,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and config.train.precision == "fp16",
    )
    batch_generator = torch.Generator().manual_seed(config.seed)
    start_step = 0
    processed_tokens = 0
    if resume is not None:
        start_step, processed_tokens = restore_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            device=device,
            batch_generator=batch_generator,
            scaler=scaler,
        )

    print(
        json.dumps(
            {
                "device": str(device),
                "run": str(output_dir),
                "train_tokens": len(train_stream),
                "validation_tokens": len(validation_stream),
                "precision": config.train.precision,
                "optimizer": config.train.optimizer,
                **model.description(),
            },
            indent=2,
        )
    )

    model.train()
    interval_start = time.perf_counter()
    interval_tokens = 0
    final_checkpoint = output_dir / "checkpoint.pt"

    for step in range(start_step + 1, config.train.max_steps + 1):
        rate = learning_rate(step, config.train)
        for group in optimizer.param_groups:
            group["lr"] = rate

        inputs, targets = train_stream.batch(
            config.train.batch_size,
            device=device,
            generator=batch_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, config.train.precision):
            logits, cross_entropy = model(inputs, targets)
            if cross_entropy is None:
                raise AssertionError("training targets did not produce a loss")
            loss = cross_entropy
            distillation_metrics: dict[str, float] = {}
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits, _ = teacher(inputs)
                auxiliary_loss, distillation_metrics = distillation_loss(
                    model,
                    teacher,
                    logits,
                    teacher_logits,
                    config,
                )
                loss = loss + auxiliary_loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}: {loss}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens = config.train.batch_size * config.model.context_length
        processed_tokens += tokens
        interval_tokens += tokens

        if step % config.train.log_interval == 0 or step == 1:
            elapsed = max(time.perf_counter() - interval_start, 1e-9)
            metric = {
                "type": "train",
                "step": step,
                "tokens": processed_tokens,
                "loss": float(loss.item()),
                "cross_entropy": float(cross_entropy.item()),
                "perplexity": math.exp(min(float(loss.item()), 20.0)),
                "learning_rate": rate,
                "gradient_norm": float(gradient_norm),
                "tokens_per_second": interval_tokens / elapsed,
                "optimizer_state_bytes": optimizer_state_bytes(optimizer),
                **distillation_metrics,
            }
            append_metric(metrics_path, metric)
            print(json.dumps(metric))
            interval_start = time.perf_counter()
            interval_tokens = 0

        if step % config.train.eval_interval == 0 or step == config.train.max_steps:
            validation = evaluate_model(
                model,
                validation_stream,
                batch_size=config.train.batch_size,
                batches=config.train.eval_batches,
                device=device,
                seed=config.seed + 10_000,
                precision=config.train.precision,
            )
            metric = {
                "type": "validation",
                "step": step,
                "tokens": processed_tokens,
                **validation,
                "weight_codes": model.quantization_stats(),
                "attention": model.attention_stats(),
                "residual": model.residual_stats(),
            }
            append_metric(metrics_path, metric)
            print(json.dumps(metric))

        if step % config.train.checkpoint_interval == 0 or step == config.train.max_steps:
            save_checkpoint(
                final_checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                processed_tokens=processed_tokens,
                batch_generator=batch_generator,
                scaler=scaler,
            )

    return final_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=VALID_MODES)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "ternary_counter", "stochastic_ternary"],
    )
    parser.add_argument("--counter-threshold", type=int)
    parser.add_argument("--transition-rate", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--population-lanes", type=int)
    parser.add_argument("--activation-levels", type=int)
    parser.add_argument("--activation-encoding", choices=VALID_ACTIVATION_ENCODINGS)
    parser.add_argument("--activation-planes", type=int)
    parser.add_argument("--activation-planes-by-layer", type=int, nargs="+")
    parser.add_argument("--residual-scale", type=float)
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
    parser.add_argument(
        "--feed-forward-activation",
        choices=VALID_FEED_FORWARD_ACTIVATIONS,
    )
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-from",
        type=Path,
        help="initialize model weights from a checkpoint but reset optimizer and step",
    )
    parser.add_argument(
        "--projection",
        type=Path,
        help="calibrated COAT projection produced by ternary-calibrate-projection",
    )
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--logit-distillation-weight", type=float)
    parser.add_argument("--attention-distillation-weight", type=float)
    parser.add_argument("--qk-distillation-weight", type=float)
    parser.add_argument("--hidden-distillation-weight", type=float)
    parser.add_argument("--distillation-temperature", type=float)
    parser.add_argument("--distillation-token-stride", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.resume and args.init_from:
        raise SystemExit("--resume and --init-from are mutually exclusive")
    config = load_config(args.config)
    if args.mode:
        config = replace_mode(config, args.mode)
    if args.device:
        config = replace(config, device=args.device)
    train_overrides = {}
    if args.precision:
        train_overrides["precision"] = args.precision
    if args.max_steps:
        train_overrides["max_steps"] = args.max_steps
    if args.batch_size:
        train_overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        train_overrides["learning_rate"] = args.learning_rate
    if args.min_learning_rate is not None:
        train_overrides["min_learning_rate"] = args.min_learning_rate
    if args.warmup_steps is not None:
        train_overrides["warmup_steps"] = args.warmup_steps
    if args.weight_decay is not None:
        train_overrides["weight_decay"] = args.weight_decay
    if args.eval_batches:
        train_overrides["eval_batches"] = args.eval_batches
    if args.output_dir:
        train_overrides["output_dir"] = str(args.output_dir)
    if args.optimizer:
        train_overrides["optimizer"] = args.optimizer
    if args.counter_threshold:
        train_overrides["counter_threshold"] = args.counter_threshold
    if args.transition_rate:
        train_overrides["transition_rate"] = args.transition_rate
    if args.logit_distillation_weight is not None:
        train_overrides["logit_distillation_weight"] = args.logit_distillation_weight
    if args.attention_distillation_weight is not None:
        train_overrides["attention_distillation_weight"] = (
            args.attention_distillation_weight
        )
    if args.qk_distillation_weight is not None:
        train_overrides["qk_distillation_weight"] = args.qk_distillation_weight
    if args.hidden_distillation_weight is not None:
        train_overrides["hidden_distillation_weight"] = args.hidden_distillation_weight
    if args.distillation_temperature is not None:
        train_overrides["distillation_temperature"] = args.distillation_temperature
    if args.distillation_token_stride is not None:
        train_overrides["distillation_token_stride"] = args.distillation_token_stride
    if args.population_lanes:
        config = replace(
            config,
            model=replace(config.model, population_lanes=args.population_lanes),
        )
    if args.activation_levels is not None:
        config = replace(
            config,
            model=replace(config.model, activation_levels=args.activation_levels),
        )
    if args.activation_encoding:
        config = replace(
            config,
            model=replace(config.model, activation_encoding=args.activation_encoding),
        )
    if args.activation_planes is not None:
        config = replace(
            config,
            model=replace(config.model, activation_planes=args.activation_planes),
        )
    if args.activation_planes_by_layer is not None:
        config = replace(
            config,
            model=replace(
                config.model,
                activation_planes_by_layer=args.activation_planes_by_layer,
            ),
        )
    if args.residual_scale:
        config = replace(
            config,
            model=replace(config.model, residual_scale=args.residual_scale),
        )
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
    if args.feed_forward_activation:
        model_overrides["feed_forward_activation"] = args.feed_forward_activation
    if model_overrides:
        config = replace(config, model=replace(config.model, **model_overrides))
    if train_overrides:
        config = replace(config, train=replace(config.train, **train_overrides))
    config.validate()
    checkpoint = train(
        config,
        run_name=args.run_name,
        resume=args.resume,
        init_from=args.init_from,
        projection=args.projection,
        teacher_checkpoint=args.teacher_checkpoint,
    )
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
