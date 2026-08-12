from __future__ import annotations

import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from ternary_llm.config import TrainConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class TokenStream:
    def __init__(self, path: str | Path, context_length: int) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"token stream not found: {self.path}. Run `uv run ternary-data prepare`."
            )
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.context_length = context_length
        if len(self.tokens) <= context_length + 1:
            raise ValueError(
                f"{self.path} has {len(self.tokens)} tokens, not enough for context "
                f"length {context_length}"
            )

    def batch(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        maximum = len(self.tokens) - self.context_length - 1
        offsets = torch.randint(maximum, (batch_size,), generator=generator).tolist()
        inputs = np.stack(
            [
                np.asarray(
                    self.tokens[offset : offset + self.context_length],
                    dtype=np.int64,
                )
                for offset in offsets
            ]
        )
        targets = np.stack(
            [
                np.asarray(
                    self.tokens[offset + 1 : offset + self.context_length + 1],
                    dtype=np.int64,
                )
                for offset in offsets
            ]
        )
        return (
            torch.from_numpy(inputs).to(device),
            torch.from_numpy(targets).to(device),
        )

    def __len__(self) -> int:
        return len(self.tokens)


def learning_rate(step: int, config: TrainConfig) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.min_learning_rate
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    ratio = (step - config.warmup_steps) / decay_steps
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_learning_rate + coefficient * (
        config.learning_rate - config.min_learning_rate
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    stream: TokenStream,
    *,
    batch_size: int,
    batches: int,
    device: torch.device,
    seed: int,
    precision: str = "fp32",
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        inputs, targets = stream.batch(
            batch_size,
            device=device,
            generator=generator,
        )
        with autocast_context(device, precision):
            _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("model did not return a validation loss")
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    mean_loss = sum(losses) / len(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
    }


@torch.no_grad()
def evaluate_model_sequential(
    model: nn.Module,
    stream: TokenStream,
    *,
    batch_size: int,
    device: torch.device,
    precision: str = "fp32",
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate deterministic, non-overlapping packed windows."""
    was_training = model.training
    model.eval()
    window = stream.context_length
    maximum_start = len(stream.tokens) - window - 1
    starts = list(range(0, maximum_start + 1, window))
    if max_batches is not None:
        starts = starts[: max_batches * batch_size]
    total_nll = 0.0
    total_tokens = 0
    for batch_start in range(0, len(starts), batch_size):
        offsets = starts[batch_start : batch_start + batch_size]
        inputs = np.stack(
            [
                np.asarray(stream.tokens[offset : offset + window], dtype=np.int64)
                for offset in offsets
            ]
        )
        targets = np.stack(
            [
                np.asarray(
                    stream.tokens[offset + 1 : offset + window + 1],
                    dtype=np.int64,
                )
                for offset in offsets
            ]
        )
        input_tensor = torch.from_numpy(inputs).to(device)
        target_tensor = torch.from_numpy(targets).to(device)
        with autocast_context(device, precision):
            _, loss = model(input_tensor, target_tensor)
        if loss is None:
            raise RuntimeError("model did not return a validation loss")
        count = target_tensor.numel()
        total_nll += float(loss.item()) * count
        total_tokens += count
    if was_training:
        model.train()
    mean_loss = total_nll / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "evaluated_tokens": float(total_tokens),
        "available_target_tokens": float(len(stream.tokens) - 1),
    }
