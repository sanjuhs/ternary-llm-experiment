from __future__ import annotations

import math
from dataclasses import asdict, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternary_llm.config import Mode, ModelConfig
from ternary_llm.projection import normalized_hadamard
from ternary_llm.quantization import (
    code_histogram,
    progressive_activation_codes,
    quantize_activation,
    quantize_attention,
    quantize_projected_activation,
    requires_coat_calibration,
    residual_refinement_activation_codes,
    ternarize_activation,
    ternarize_activation_with_learned_scale,
    ternarize_weight,
    ternary_activation_codes,
    uses_activation_projection,
    uses_quantized_activations,
    uses_ternary_weights,
)


class TernaryLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        mode: Mode,
        weight_threshold: float,
        activation_threshold: float,
        activation_levels: int,
        activation_encoding: str,
        activation_planes: int,
        population_lanes: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mode = mode
        self.weight_threshold = weight_threshold
        self.activation_threshold = activation_threshold
        self.activation_levels = activation_levels
        self.activation_encoding = activation_encoding
        self.activation_planes = activation_planes
        self.population_lanes = population_lanes
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        if uses_quantized_activations(self.mode):
            inputs = quantize_activation(
                inputs,
                self.mode,
                self.activation_threshold,
                lanes=self.population_lanes,
                levels=self.activation_levels,
                encoding=self.activation_encoding,
                planes=self.activation_planes,
            )
        weight = self.weight
        if uses_ternary_weights(self.mode):
            weight = ternarize_weight(weight, self.weight_threshold)
        return F.linear(inputs, weight)


class TernaryRMSNorm(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        mode: Mode,
        weight_threshold: float,
        activation_threshold: float,
        activation_levels: int,
        activation_encoding: str,
        activation_planes: int,
        population_lanes: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.mode = mode
        self.weight_threshold = weight_threshold
        self.activation_threshold = activation_threshold
        self.activation_levels = activation_levels
        self.activation_encoding = activation_encoding
        self.activation_planes = activation_planes
        self.population_lanes = population_lanes
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + self.eps)
        weight = self.weight
        if uses_ternary_weights(self.mode):
            weight = ternarize_weight(weight, self.weight_threshold)
        output = normalized * weight
        if uses_quantized_activations(self.mode):
            output = quantize_activation(
                output,
                self.mode,
                self.activation_threshold,
                lanes=self.population_lanes,
                levels=self.activation_levels,
                encoding=self.activation_encoding,
                planes=self.activation_planes,
            )
        return output


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, mode: Mode) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.dropout = config.dropout
        linear_args = {
            "mode": mode,
            "weight_threshold": config.weight_threshold,
            "activation_threshold": config.activation_threshold,
            "activation_levels": config.activation_levels,
            "activation_encoding": config.activation_encoding,
            "activation_planes": config.activation_planes,
            "population_lanes": config.population_lanes if mode == "population_ternary" else 1,
        }
        self.qkv = TernaryLinear(config.d_model, 3 * config.d_model, **linear_args)
        self.projection = TernaryLinear(config.d_model, config.d_model, **linear_args)
        self.mode = mode
        self.weight_threshold = config.weight_threshold
        self.activation_threshold = config.activation_threshold
        self.activation_levels = config.activation_levels
        self.activation_encoding = config.activation_encoding
        self.activation_planes = config.activation_planes
        self.population_lanes = config.population_lanes if mode == "population_ternary" else 1
        self.attention_quantization = config.attention_quantization
        self.attention_clip = config.attention_clip
        self.attention_threshold = config.attention_threshold
        self.qkv_quantization = config.qkv_quantization
        self.qkv_scale_granularity = config.qkv_scale_granularity
        self.attention_rectification = config.attention_rectification
        self.attention_gate = config.attention_gate
        if (
            self.qkv_quantization == "ternary"
            and self.qkv_scale_granularity == "learned_head"
        ):
            self.qkv_log_scales = nn.Parameter(
                torch.full(
                    (3, self.n_heads),
                    math.log(config.qkv_scale_initial),
                )
            )
        else:
            self.register_parameter("qkv_log_scales", None)
        if self.attention_rectification == "qvit":
            parameter_shape = (1, self.n_heads, 1, self.head_size)
            self.q_gamma = nn.Parameter(torch.ones(parameter_shape))
            self.q_beta = nn.Parameter(torch.zeros(parameter_shape))
            self.k_gamma = nn.Parameter(torch.ones(parameter_shape))
            self.k_beta = nn.Parameter(torch.zeros(parameter_shape))
        else:
            self.register_parameter("q_gamma", None)
            self.register_parameter("q_beta", None)
            self.register_parameter("k_gamma", None)
            self.register_parameter("k_beta", None)
        if self.attention_gate != "none":
            self.gate_weight = nn.Parameter(torch.zeros(self.n_heads, self.head_size))
            initial_logit = math.log(
                config.attention_gate_initial / (1.0 - config.attention_gate_initial)
            )
            self.gate_bias = nn.Parameter(torch.full((self.n_heads,), initial_logit))
        else:
            self.register_parameter("gate_weight", None)
            self.register_parameter("gate_bias", None)
        self.last_attention_stats: dict[str, float] = {}
        self.capture_distillation = False
        self.last_q: Tensor | None = None
        self.last_k: Tensor | None = None
        self.last_probabilities: Tensor | None = None
        self.last_qkv_scale_stats: dict[str, float] = {}

    @staticmethod
    def _standardize(tensor: Tensor) -> Tensor:
        centered = tensor - tensor.mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(
            centered.square().mean(dim=-1, keepdim=True) + 1e-5
        )

    def _rectify_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        if self.attention_rectification != "qvit":
            return q, k
        if any(
            parameter is None
            for parameter in (self.q_gamma, self.q_beta, self.k_gamma, self.k_beta)
        ):
            raise AssertionError("Q-ViT rectification parameters are missing")
        q = self._standardize(q) * self.q_gamma + self.q_beta
        k = self._standardize(k) * self.k_gamma + self.k_beta
        return q, k

    def _quantize_qkv(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        code_tensors: dict[str, Tensor] = {}
        if self.qkv_quantization == "ternary":
            if self.qkv_scale_granularity == "learned_head":
                if self.qkv_log_scales is None:
                    raise AssertionError("learned QKV scales are missing")
                quantized_tensors = []
                scale_stats = {}
                for index, (name, tensor) in enumerate((("q", q), ("k", k), ("v", v))):
                    scale = self.qkv_log_scales[index].exp().view(
                        1,
                        self.n_heads,
                        1,
                        1,
                    )
                    quantized, codes = ternarize_activation_with_learned_scale(
                        tensor,
                        scale,
                        self.activation_threshold,
                    )
                    quantized_tensors.append(quantized)
                    code_tensors[name] = codes
                    detached_scale = scale.detach()
                    scale_stats[f"{name}_scale_mean"] = float(
                        detached_scale.mean().item()
                    )
                    scale_stats[f"{name}_scale_min"] = float(
                        detached_scale.min().item()
                    )
                    scale_stats[f"{name}_scale_max"] = float(
                        detached_scale.max().item()
                    )
                self.last_qkv_scale_stats = scale_stats
                return (*quantized_tensors, code_tensors)
            for name, tensor in (("q", q), ("k", k), ("v", v)):
                codes, _ = ternary_activation_codes(
                    tensor,
                    self.activation_threshold,
                )
                code_tensors[name] = codes
            self.last_qkv_scale_stats = {}
            return (
                ternarize_activation(q, self.activation_threshold),
                ternarize_activation(k, self.activation_threshold),
                ternarize_activation(v, self.activation_threshold),
                code_tensors,
            )
        if uses_quantized_activations(self.mode):
            q = quantize_activation(
                q,
                self.mode,
                self.activation_threshold,
                lanes=self.population_lanes,
                levels=self.activation_levels,
                encoding=self.activation_encoding,
                planes=self.activation_planes,
            )
            k = quantize_activation(
                k,
                self.mode,
                self.activation_threshold,
                lanes=self.population_lanes,
                levels=self.activation_levels,
                encoding=self.activation_encoding,
                planes=self.activation_planes,
            )
            v = quantize_activation(
                v,
                self.mode,
                self.activation_threshold,
                lanes=self.population_lanes,
                levels=self.activation_levels,
                encoding=self.activation_encoding,
                planes=self.activation_planes,
            )
        return q, k, v, code_tensors

    def _attention_gate(self, inputs: Tensor) -> tuple[Tensor | None, Tensor | None]:
        if self.attention_gate == "none":
            return None, None
        if self.gate_weight is None or self.gate_bias is None:
            raise AssertionError("attention gate parameters are missing")
        batch, length, _ = inputs.shape
        head_inputs = inputs.view(
            batch,
            length,
            self.n_heads,
            self.head_size,
        ).permute(0, 2, 1, 3)
        gate_inputs = ternarize_activation(head_inputs, self.activation_threshold)
        gate_weight = ternarize_weight(self.gate_weight, self.weight_threshold)
        logits = (gate_inputs * gate_weight.view(1, self.n_heads, 1, -1)).sum(dim=-1)
        logits = logits + self.gate_bias.view(1, self.n_heads, 1)
        probabilities = torch.sigmoid(logits)
        if self.attention_gate == "binary":
            codes = (probabilities.detach() >= 0.5).to(probabilities.dtype)
            probabilities = probabilities + (codes - probabilities).detach()
            return probabilities, codes
        return probabilities, None

    def forward(self, inputs: Tensor) -> Tensor:
        batch, length, channels = inputs.shape
        qkv = self.qkv(inputs)
        qkv = qkv.view(batch, length, 3, self.n_heads, self.head_size)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        q, k = self._rectify_qk(q, k)
        q, k, v, qkv_codes = self._quantize_qkv(q, k, v)

        scores = q @ k.transpose(-2, -1) / self.head_size**0.5
        valid = torch.ones(
            (length, length),
            device=scores.device,
            dtype=torch.bool,
        ).tril()
        probabilities, score_codes, probability_codes = quantize_attention(
            scores,
            valid,
            scheme=self.attention_quantization,
            clip=self.attention_clip,
            threshold=self.attention_threshold,
        )
        gate, gate_codes = self._attention_gate(inputs)
        if self.capture_distillation:
            self.last_q = q
            self.last_k = k
            self.last_probabilities = probabilities
        if not self.training:
            diagnostic_mask = valid.to(torch.float32).view(1, 1, length, length)
            diagnostic_count = diagnostic_mask.sum() * batch * self.n_heads
            with torch.no_grad():
                entropy = -(
                    probabilities.clamp_min(1e-8)
                    * probabilities.clamp_min(1e-8).log()
                ).sum(dim=-1)
                stats = {
                    "entropy": float(entropy.mean().item()),
                    "zero_fraction": float(
                        (
                            (probabilities == 0).to(torch.float32)
                            * diagnostic_mask
                        ).sum().item()
                        / diagnostic_count.item()
                    ),
                }
                if score_codes is not None:
                    for code in range(-3, 1):
                        stats[f"score_code_{code}_fraction"] = float(
                            (
                                (score_codes == code).to(torch.float32)
                                * diagnostic_mask
                            ).sum().item()
                            / diagnostic_count.item()
                        )
                if probability_codes is not None:
                    maximum = (
                        1
                        if self.attention_quantization
                        in {"prob_binary", "score_lut_prob_binary"}
                        else 3
                    )
                    for code in range(maximum + 1):
                        stats[f"probability_code_{code}_fraction"] = float(
                            (
                                (probability_codes == code).to(torch.float32)
                                * diagnostic_mask
                            ).sum().item()
                                / diagnostic_count.item()
                        )
                for name, codes in qkv_codes.items():
                    for code, label in ((-1, "negative"), (0, "zero"), (1, "positive")):
                        stats[f"{name}_{label}_fraction"] = float(
                            (codes == code).to(torch.float32).mean().item()
                        )
                stats.update(self.last_qkv_scale_stats)
                if gate is not None:
                    stats["gate_mean"] = float(gate.mean().item())
                if gate_codes is not None:
                    stats["gate_open_fraction"] = float(gate_codes.mean().item())
                self.last_attention_stats = stats
        probabilities = F.dropout(
            probabilities,
            p=self.dropout,
            training=self.training,
        )
        attended = probabilities @ v
        if gate is not None:
            attended = attended * gate.unsqueeze(-1)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        output = self.projection(attended)
        if uses_quantized_activations(self.mode):
            output = quantize_activation(
                output,
                self.mode,
                self.activation_threshold,
                lanes=self.population_lanes,
                levels=self.activation_levels,
                encoding=self.activation_encoding,
                planes=self.activation_planes,
            )
        return output


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, mode: Mode) -> None:
        super().__init__()
        hidden_size = config.ff_multiplier * config.d_model
        linear_args = {
            "mode": mode,
            "weight_threshold": config.weight_threshold,
            "activation_threshold": config.activation_threshold,
            "activation_levels": config.activation_levels,
            "activation_encoding": config.activation_encoding,
            "activation_planes": config.activation_planes,
            "population_lanes": config.population_lanes if mode == "population_ternary" else 1,
        }
        self.up = TernaryLinear(config.d_model, hidden_size, **linear_args)
        self.down = TernaryLinear(hidden_size, config.d_model, **linear_args)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = mode
        self.activation_threshold = config.activation_threshold
        self.activation_levels = config.activation_levels
        self.activation_encoding = config.activation_encoding
        self.activation_planes = config.activation_planes
        self.population_lanes = config.population_lanes if mode == "population_ternary" else 1
        self.feed_forward_activation = config.feed_forward_activation

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.up(inputs)
        if self.feed_forward_activation == "relu":
            hidden = F.relu(hidden)
        else:
            hidden = F.gelu(hidden, approximate="tanh")
        if uses_quantized_activations(self.mode):
            hidden = quantize_activation(
                hidden,
                self.mode,
                self.activation_threshold,
                lanes=self.population_lanes,
                levels=self.activation_levels,
                encoding=self.activation_encoding,
                planes=self.activation_planes,
            )
        output = self.down(hidden)
        output = self.dropout(output)
        if uses_quantized_activations(self.mode):
            output = quantize_activation(
                output,
                self.mode,
                self.activation_threshold,
                lanes=self.population_lanes,
                levels=self.activation_levels,
                encoding=self.activation_encoding,
                planes=self.activation_planes,
            )
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, mode: Mode) -> None:
        super().__init__()
        norm_args = {
            "mode": mode,
            "weight_threshold": config.weight_threshold,
            "activation_threshold": config.activation_threshold,
            "activation_levels": config.activation_levels,
            "activation_encoding": config.activation_encoding,
            "activation_planes": config.activation_planes,
            "population_lanes": config.population_lanes if mode == "population_ternary" else 1,
        }
        self.attention_norm = TernaryRMSNorm(config.d_model, **norm_args)
        self.attention = CausalSelfAttention(config, mode)
        self.feed_forward_norm = TernaryRMSNorm(config.d_model, **norm_args)
        self.feed_forward = FeedForward(config, mode)
        self.mode = mode
        self.activation_threshold = config.activation_threshold
        self.activation_levels = config.activation_levels
        self.activation_encoding = config.activation_encoding
        self.activation_planes = config.activation_planes
        self.population_lanes = config.population_lanes if mode == "population_ternary" else 1
        self.residual_scale = config.residual_scale
        projection = normalized_hadamard(config.d_model)
        self.register_buffer("activation_projection", projection, persistent=True)
        self.register_buffer(
            "coat_projection_ready",
            torch.tensor(not requires_coat_calibration(mode)),
            persistent=True,
        )
        self.last_residual_stats: dict[str, float] = {}
        self.capture_distillation = False
        self.last_hidden: Tensor | None = None

    def set_activation_projection(self, projection: Tensor) -> None:
        expected = (self.activation_projection.shape[0],) * 2
        if tuple(projection.shape) != expected:
            raise ValueError(
                f"projection shape must be {expected}, got {tuple(projection.shape)}"
            )
        self.activation_projection.copy_(projection)
        self.coat_projection_ready.fill_(True)

    def _quantize_residual(self, tensor: Tensor) -> Tensor:
        if requires_coat_calibration(self.mode) and not bool(self.coat_projection_ready):
            raise RuntimeError("COAT mode requires a calibrated activation projection")
        if uses_activation_projection(self.mode):
            if self.mode in {"coat_progressive", "hadamard_progressive"} and not self.training:
                q = self.activation_projection.to(
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                rotated = tensor @ q
                if self.activation_encoding == "uniform":
                    codes, _ = progressive_activation_codes(
                        rotated,
                        self.activation_levels,
                        threshold=self.activation_threshold,
                    )
                    qmax = (self.activation_levels - 1) // 2
                    self.last_residual_stats = {
                        "configured_levels": float(self.activation_levels),
                        "used_levels": float(codes.unique().numel()),
                        "zero_fraction": float(
                            (codes == 0).to(torch.float32).mean().item()
                        ),
                        "saturation_fraction": float(
                            (codes.abs() == qmax).to(torch.float32).mean().item()
                        ),
                    }
                else:
                    binary = self.activation_encoding == "residual_binary"
                    codes, scales = residual_refinement_activation_codes(
                        rotated,
                        self.activation_planes,
                        binary=binary,
                        threshold=self.activation_threshold,
                    )
                    reconstructed = (codes * scales).sum(dim=-1)
                    error = (rotated - reconstructed).square().mean()
                    energy = rotated.square().mean().clamp_min(1e-8)
                    stats = {
                        "planes": float(self.activation_planes),
                        "logical_bits_per_scalar": float(
                            self.activation_planes
                            if binary
                            else self.activation_planes * math.log2(3.0)
                        ),
                        "normalized_mse": float((error / energy).item()),
                        "zero_fraction": float(
                            (codes == 0).to(torch.float32).mean().item()
                        ),
                    }
                    for plane in range(self.activation_planes):
                        plane_codes = codes[..., plane]
                        stats[f"plane_{plane}_zero_fraction"] = float(
                            (plane_codes == 0).to(torch.float32).mean().item()
                        )
                    self.last_residual_stats = stats
            return quantize_projected_activation(
                tensor,
                self.mode,
                self.activation_threshold,
                self.activation_projection,
                lanes=self.population_lanes,
                levels=self.activation_levels,
                encoding=self.activation_encoding,
                planes=self.activation_planes,
            )
        return quantize_activation(
            tensor,
            self.mode,
            self.activation_threshold,
            lanes=self.population_lanes,
            levels=self.activation_levels,
            encoding=self.activation_encoding,
            planes=self.activation_planes,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.residual_scale * self.attention(self.attention_norm(inputs))
        if uses_quantized_activations(self.mode):
            hidden = self._quantize_residual(hidden)
        hidden = hidden + self.residual_scale * self.feed_forward(
            self.feed_forward_norm(hidden)
        )
        if uses_quantized_activations(self.mode):
            hidden = self._quantize_residual(hidden)
        if self.capture_distillation:
            self.last_hidden = hidden
        return hidden


class TernaryGPT(nn.Module):
    def __init__(self, config: ModelConfig, mode: Mode = "float") -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.mode = mode
        self.token_embedding = nn.Parameter(torch.empty(config.vocab_size, config.d_model))
        self.position_embedding = nn.Parameter(torch.empty(config.context_length, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                (
                    replace(
                        config,
                        activation_planes=config.activation_planes_by_layer[layer],
                    )
                    if config.activation_planes_by_layer is not None
                    else config
                ),
                mode,
            )
            for layer in range(config.n_layers)
        )
        self.final_norm = TernaryRMSNorm(
            config.d_model,
            mode=mode,
            weight_threshold=config.weight_threshold,
            activation_threshold=config.activation_threshold,
            activation_levels=config.activation_levels,
            activation_encoding=config.activation_encoding,
            activation_planes=config.activation_planes,
            population_lanes=(
                config.population_lanes if mode == "population_ternary" else 1
            ),
        )
        nn.init.normal_(self.token_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.apply(self._initialize)

    def set_activation_projection(self, projection: Tensor) -> None:
        for block in self.blocks:
            block.set_activation_projection(projection)

    def reset_hadamard_projection(self) -> None:
        projection = normalized_hadamard(self.config.d_model)
        self.set_activation_projection(projection)

    def load_activation_projection(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        projection = payload["projection"] if isinstance(payload, dict) else payload
        self.set_activation_projection(projection)

    def set_capture_distillation(self, enabled: bool) -> None:
        for block in self.blocks:
            block.attention.capture_distillation = enabled
            block.capture_distillation = enabled

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, TernaryLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embedding_weight(self) -> Tensor:
        if uses_ternary_weights(self.mode):
            return ternarize_weight(self.token_embedding, self.config.weight_threshold)
        return self.token_embedding

    def _position_weight(self) -> Tensor:
        if uses_ternary_weights(self.mode):
            return ternarize_weight(self.position_embedding, self.config.weight_threshold)
        return self.position_embedding

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        _, length = token_ids.shape
        if length > self.config.context_length:
            raise ValueError(
                f"sequence length {length} exceeds context {self.config.context_length}"
            )
        positions = torch.arange(length, device=token_ids.device)
        hidden = F.embedding(token_ids, self._embedding_weight())
        hidden = hidden + F.embedding(positions, self._position_weight())
        hidden = self.dropout(hidden)
        if uses_quantized_activations(self.mode):
            projection = self.blocks[0].activation_projection
            hidden = quantize_projected_activation(
                hidden,
                self.mode,
                self.config.activation_threshold,
                projection,
                lanes=(
                    self.config.population_lanes if self.mode == "population_ternary" else 1
                ),
                levels=self.config.activation_levels,
                encoding=self.config.activation_encoding,
                planes=self.config.activation_planes,
            )

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.final_norm(hidden)
        if uses_quantized_activations(self.mode):
            projection = self.blocks[0].activation_projection
            hidden = quantize_projected_activation(
                hidden,
                self.mode,
                self.config.activation_threshold,
                projection,
                lanes=(
                    self.config.population_lanes if self.mode == "population_ternary" else 1
                ),
                levels=self.config.activation_levels,
                encoding=self.config.activation_encoding,
                planes=self.config.activation_planes,
            )
        logits = F.linear(hidden, self._embedding_weight())
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_id: int | None = None,
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, next_logits.size(-1))
                cutoff = torch.topk(next_logits, k).values[:, [-1]]
                next_logits = next_logits.masked_fill(next_logits < cutoff, -torch.inf)
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
            if eos_id is not None and bool(torch.all(next_token == eos_id)):
                break
        return token_ids

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def quantization_stats(self) -> dict[str, Any]:
        if not uses_ternary_weights(self.mode):
            return {"enabled": False}
        counts = {"-1": 0, "0": 0, "+1": 0}
        for parameter in self.parameters():
            histogram = code_histogram(parameter, self.config.weight_threshold)
            for code, count in histogram.items():
                counts[code] += count
        total = sum(counts.values())
        fractions = {key: value / total for key, value in counts.items()}
        return {"enabled": True, "counts": counts, "fractions": fractions}

    def attention_stats(self) -> dict[str, Any]:
        per_layer = [
            {"layer": index, **block.attention.last_attention_stats}
            for index, block in enumerate(self.blocks)
            if block.attention.last_attention_stats
        ]
        aggregate: dict[str, float] = {}
        if per_layer:
            keys = set.intersection(*(set(layer) for layer in per_layer)) - {"layer"}
            aggregate = {
                key: sum(float(layer[key]) for layer in per_layer) / len(per_layer)
                for key in sorted(keys)
            }
        return {
            "scheme": self.config.attention_quantization,
            "clip": self.config.attention_clip,
            "threshold": self.config.attention_threshold,
            "qkv_quantization": self.config.qkv_quantization,
            "rectification": self.config.attention_rectification,
            "gate": self.config.attention_gate,
            "aggregate": aggregate,
            "layers": per_layer,
        }

    def residual_stats(self) -> dict[str, Any]:
        per_layer = [
            {"layer": index, **block.last_residual_stats}
            for index, block in enumerate(self.blocks)
            if block.last_residual_stats
        ]
        aggregate: dict[str, float] = {}
        if per_layer:
            keys = set.intersection(*(set(layer) for layer in per_layer)) - {"layer"}
            aggregate = {
                key: sum(float(layer[key]) for layer in per_layer) / len(per_layer)
                for key in sorted(keys)
            }
        return {"aggregate": aggregate, "layers": per_layer}

    def description(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "parameters": self.parameter_count(),
            "model": asdict(self.config),
            "quantization": self.quantization_stats(),
            "attention": {
                "scheme": self.config.attention_quantization,
                "clip": self.config.attention_clip,
                "threshold": self.config.attention_threshold,
                "qkv_quantization": self.config.qkv_quantization,
                "rectification": self.config.attention_rectification,
                "gate": self.config.attention_gate,
            },
            "residual": {
                "levels": self.config.activation_levels
                if self.mode in {"coat_progressive", "hadamard_progressive"}
                else None,
                "encoding": self.config.activation_encoding,
                "planes": self.config.activation_planes,
            },
        }
