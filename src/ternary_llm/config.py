from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Mode = Literal[
    "float",
    "ternary_weights",
    "ternary_activations",
    "ternary_forward",
    "population_ternary",
    "hadamard_a4",
    "coat_a4",
    "hadamard_ternary",
    "coat_ternary",
]
VALID_MODES: tuple[Mode, ...] = (
    "float",
    "ternary_weights",
    "ternary_activations",
    "ternary_forward",
    "population_ternary",
    "hadamard_a4",
    "coat_a4",
    "hadamard_ternary",
    "coat_ternary",
)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    ff_multiplier: int = 4
    dropout: float = 0.0
    activation_threshold: float = 0.5
    weight_threshold: float = 0.5
    population_lanes: int = 1
    residual_scale: float = 1.0

    def validate(self) -> None:
        if self.vocab_size <= 4:
            raise ValueError("vocab_size must be greater than the four special tokens")
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.ff_multiplier < 1:
            raise ValueError("ff_multiplier must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation_threshold <= 0 or self.weight_threshold <= 0:
            raise ValueError("ternary thresholds must be positive")
        if self.population_lanes < 1:
            raise ValueError("population_lanes must be positive")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")


@dataclass(frozen=True)
class DataConfig:
    train_bin: str
    validation_bin: str
    tokenizer: str


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    max_steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int
    output_dir: str
    log_interval: int = 10
    precision: str = "fp32"
    optimizer: str = "adamw"
    counter_threshold: int = 8
    transition_rate: float = 0.002

    def validate(self) -> None:
        integer_fields = {
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be non-negative, with a positive maximum")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight_decay must be non-negative and grad_clip positive")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        if self.optimizer not in {"adamw", "ternary_counter", "stochastic_ternary"}:
            raise ValueError(
                "optimizer must be adamw, ternary_counter, or stochastic_ternary"
            )
        if not 1 <= self.counter_threshold <= 127:
            raise ValueError("counter_threshold must be between 1 and 127")
        if not 0.0 < self.transition_rate <= 1.0:
            raise ValueError("transition_rate must be in (0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    mode: Mode
    device: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps, or cuda")
        self.model.validate()
        self.train.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    config = ExperimentConfig(
        seed=int(raw["seed"]),
        mode=raw["mode"],
        device=raw.get("device", "auto"),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        train=TrainConfig(**raw["train"]),
    )
    config.validate()
    return config


def replace_mode(config: ExperimentConfig, mode: str) -> ExperimentConfig:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    updated = ExperimentConfig(
        seed=config.seed,
        mode=mode,  # type: ignore[arg-type]
        device=config.device,
        model=config.model,
        data=config.data,
        train=config.train,
    )
    updated.validate()
    return updated
