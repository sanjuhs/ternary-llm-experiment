from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

AttentionQuantization = Literal[
    "float",
    "score_int2",
    "score_int2_lut",
    "prob_int2",
    "score_prob_int2",
    "score_lut_prob_int2",
    "prob_binary",
    "score_lut_prob_binary",
]
VALID_ATTENTION_QUANTIZATIONS: tuple[AttentionQuantization, ...] = (
    "float",
    "score_int2",
    "score_int2_lut",
    "prob_int2",
    "score_prob_int2",
    "score_lut_prob_int2",
    "prob_binary",
    "score_lut_prob_binary",
)

AttentionNormalization = Literal["softmax", "softmax1"]
VALID_ATTENTION_NORMALIZATIONS: tuple[AttentionNormalization, ...] = (
    "softmax",
    "softmax1",
)

QKVQuantization = Literal["inherit", "ternary", "binary_qk_ternary_v"]
VALID_QKV_QUANTIZATIONS: tuple[QKVQuantization, ...] = (
    "inherit",
    "ternary",
    "binary_qk_ternary_v",
)

QKVScaleGranularity = Literal["token", "learned_head"]
VALID_QKV_SCALE_GRANULARITIES: tuple[QKVScaleGranularity, ...] = (
    "token",
    "learned_head",
)

AttentionRectification = Literal["none", "qvit"]
VALID_ATTENTION_RECTIFICATIONS: tuple[AttentionRectification, ...] = ("none", "qvit")

AttentionGate = Literal["none", "sigmoid", "binary"]
VALID_ATTENTION_GATES: tuple[AttentionGate, ...] = ("none", "sigmoid", "binary")

FeedForwardActivation = Literal["gelu", "relu"]
VALID_FEED_FORWARD_ACTIVATIONS: tuple[FeedForwardActivation, ...] = ("gelu", "relu")

RMSNormQuantization = Literal["float", "integer_reference"]
VALID_RMS_NORM_QUANTIZATIONS: tuple[RMSNormQuantization, ...] = (
    "float",
    "integer_reference",
)

ActivationEncoding = Literal["uniform", "residual_binary", "residual_ternary"]
VALID_ACTIVATION_ENCODINGS: tuple[ActivationEncoding, ...] = (
    "uniform",
    "residual_binary",
    "residual_ternary",
)

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
    "coat_progressive",
    "hadamard_progressive",
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
    "coat_progressive",
    "hadamard_progressive",
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
    activation_levels: int = 19
    activation_encoding: ActivationEncoding = "uniform"
    activation_planes: int = 1
    activation_planes_by_layer: list[int] | None = None
    weight_threshold: float = 0.5
    population_lanes: int = 1
    residual_scale: float = 1.0
    attention_quantization: AttentionQuantization = "float"
    attention_normalization: AttentionNormalization = "softmax"
    attention_clip: float = 6.0
    attention_threshold: float = 0.5
    qkv_quantization: QKVQuantization = "inherit"
    qkv_scale_granularity: QKVScaleGranularity = "token"
    qkv_scale_initial: float = 0.5
    attention_rectification: AttentionRectification = "none"
    attention_gate: AttentionGate = "none"
    attention_gate_initial: float = 0.9
    feed_forward_activation: FeedForwardActivation = "gelu"
    rms_norm_quantization: RMSNormQuantization = "float"

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
        if self.activation_levels < 3 or self.activation_levels % 2 == 0:
            raise ValueError("activation_levels must be an odd integer of at least 3")
        if self.activation_encoding not in VALID_ACTIVATION_ENCODINGS:
            raise ValueError(
                "activation_encoding must be one of "
                f"{VALID_ACTIVATION_ENCODINGS}, got {self.activation_encoding!r}"
            )
        if self.activation_planes < 1:
            raise ValueError("activation_planes must be positive")
        if self.activation_planes_by_layer is not None:
            if len(self.activation_planes_by_layer) != self.n_layers:
                raise ValueError(
                    "activation_planes_by_layer must contain one value per layer"
                )
            if any(planes < 1 for planes in self.activation_planes_by_layer):
                raise ValueError(
                    "activation_planes_by_layer values must all be positive"
                )
        if self.population_lanes < 1:
            raise ValueError("population_lanes must be positive")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if self.attention_quantization not in VALID_ATTENTION_QUANTIZATIONS:
            raise ValueError(
                "attention_quantization must be one of "
                f"{VALID_ATTENTION_QUANTIZATIONS}, got {self.attention_quantization!r}"
            )
        if self.attention_normalization not in VALID_ATTENTION_NORMALIZATIONS:
            raise ValueError(
                "attention_normalization must be one of "
                f"{VALID_ATTENTION_NORMALIZATIONS}, "
                f"got {self.attention_normalization!r}"
            )
        if self.attention_clip <= 0:
            raise ValueError("attention_clip must be positive")
        if not 0.0 < self.attention_threshold <= 1.0:
            raise ValueError("attention_threshold must be in (0, 1]")
        if self.qkv_quantization not in VALID_QKV_QUANTIZATIONS:
            raise ValueError(
                f"qkv_quantization must be one of {VALID_QKV_QUANTIZATIONS}, "
                f"got {self.qkv_quantization!r}"
            )
        if self.qkv_scale_granularity not in VALID_QKV_SCALE_GRANULARITIES:
            raise ValueError(
                "qkv_scale_granularity must be one of "
                f"{VALID_QKV_SCALE_GRANULARITIES}, "
                f"got {self.qkv_scale_granularity!r}"
            )
        if (
            self.qkv_scale_granularity != "token"
            and self.qkv_quantization
            not in {"ternary", "binary_qk_ternary_v"}
        ):
            raise ValueError(
                "non-token QKV scales require quantized Q/K/V"
            )
        if self.qkv_scale_initial <= 0:
            raise ValueError("qkv_scale_initial must be positive")
        if self.attention_rectification not in VALID_ATTENTION_RECTIFICATIONS:
            raise ValueError(
                "attention_rectification must be one of "
                f"{VALID_ATTENTION_RECTIFICATIONS}, "
                f"got {self.attention_rectification!r}"
            )
        if self.attention_gate not in VALID_ATTENTION_GATES:
            raise ValueError(
                f"attention_gate must be one of {VALID_ATTENTION_GATES}, "
                f"got {self.attention_gate!r}"
            )
        if not 0.0 < self.attention_gate_initial < 1.0:
            raise ValueError("attention_gate_initial must be in (0, 1)")
        if self.feed_forward_activation not in VALID_FEED_FORWARD_ACTIVATIONS:
            raise ValueError(
                "feed_forward_activation must be one of "
                f"{VALID_FEED_FORWARD_ACTIVATIONS}, "
                f"got {self.feed_forward_activation!r}"
            )
        if self.rms_norm_quantization not in VALID_RMS_NORM_QUANTIZATIONS:
            raise ValueError(
                "rms_norm_quantization must be one of "
                f"{VALID_RMS_NORM_QUANTIZATIONS}, "
                f"got {self.rms_norm_quantization!r}"
            )


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
    logit_distillation_weight: float = 0.0
    attention_distillation_weight: float = 0.0
    qk_distillation_weight: float = 0.0
    hidden_distillation_weight: float = 0.0
    distillation_temperature: float = 1.0
    distillation_token_stride: int = 4

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
        distillation_weights = (
            self.logit_distillation_weight,
            self.attention_distillation_weight,
            self.qk_distillation_weight,
            self.hidden_distillation_weight,
        )
        if any(weight < 0 for weight in distillation_weights):
            raise ValueError("distillation weights must be non-negative")
        if self.distillation_temperature <= 0:
            raise ValueError("distillation_temperature must be positive")
        if self.distillation_token_stride < 1:
            raise ValueError("distillation_token_stride must be positive")


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
