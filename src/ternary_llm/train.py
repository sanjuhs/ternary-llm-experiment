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
from tokenizers import Tokenizer

from ternary_llm.config import VALID_MODES, ExperimentConfig, load_config, replace_mode
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


def train(
    config: ExperimentConfig,
    *,
    run_name: str | None = None,
    resume: Path | None = None,
    init_from: Path | None = None,
    projection: Path | None = None,
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
    if init_from is not None:
        initial = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(initial["model"], strict=False)
    if projection is not None:
        model.load_activation_projection(str(projection))
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
            _, loss = model(inputs, targets)
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
                "perplexity": math.exp(min(float(loss.item()), 20.0)),
                "learning_rate": rate,
                "gradient_norm": float(gradient_norm),
                "tokens_per_second": interval_tokens / elapsed,
                "optimizer_state_bytes": optimizer_state_bytes(optimizer),
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
    parser.add_argument("--population-lanes", type=int)
    parser.add_argument("--residual-scale", type=float)
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
    if args.population_lanes:
        config = replace(
            config,
            model=replace(config.model, population_lanes=args.population_lanes),
        )
    if args.residual_scale:
        config = replace(
            config,
            model=replace(config.model, residual_scale=args.residual_scale),
        )
    if train_overrides:
        config = replace(config, train=replace(config.train, **train_overrides))
    config.validate()
    checkpoint = train(
        config,
        run_name=args.run_name,
        resume=args.resume,
        init_from=args.init_from,
        projection=args.projection,
    )
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
