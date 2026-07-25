from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternary_llm.config import Mode, ModelConfig
from ternary_llm.projection import normalized_hadamard
from ternary_llm.quantization import (
    code_histogram,
    quantize_activation,
    quantize_attention,
    quantize_projected_activation,
    requires_coat_calibration,
    ternarize_weight,
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
        population_lanes: int,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mode = mode
        self.weight_threshold = weight_threshold
        self.activation_threshold = activation_threshold
        self.population_lanes = population_lanes
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        if uses_quantized_activations(self.mode):
            inputs = quantize_activation(
                inputs, self.mode, self.activation_threshold, lanes=self.population_lanes
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
        population_lanes: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.mode = mode
        self.weight_threshold = weight_threshold
        self.activation_threshold = activation_threshold
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
                output, self.mode, self.activation_threshold, lanes=self.population_lanes
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
            "population_lanes": config.population_lanes if mode == "population_ternary" else 1,
        }
        self.qkv = TernaryLinear(config.d_model, 3 * config.d_model, **linear_args)
        self.projection = TernaryLinear(config.d_model, config.d_model, **linear_args)
        self.mode = mode
        self.activation_threshold = config.activation_threshold
        self.population_lanes = config.population_lanes if mode == "population_ternary" else 1
        self.attention_quantization = config.attention_quantization
        self.attention_clip = config.attention_clip
        self.attention_threshold = config.attention_threshold
        self.last_attention_stats: dict[str, float] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        batch, length, channels = inputs.shape
        qkv = self.qkv(inputs)
        qkv = qkv.view(batch, length, 3, self.n_heads, self.head_size)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        if uses_quantized_activations(self.mode):
            q = quantize_activation(
                q, self.mode, self.activation_threshold, lanes=self.population_lanes
            )
            k = quantize_activation(
                k, self.mode, self.activation_threshold, lanes=self.population_lanes
            )
            v = quantize_activation(
                v, self.mode, self.activation_threshold, lanes=self.population_lanes
            )

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
                    maximum = 1 if self.attention_quantization == "prob_binary" else 3
                    for code in range(maximum + 1):
                        stats[f"probability_code_{code}_fraction"] = float(
                            (
                                (probability_codes == code).to(torch.float32)
                                * diagnostic_mask
                            ).sum().item()
                            / diagnostic_count.item()
                        )
                self.last_attention_stats = stats
        probabilities = F.dropout(
            probabilities,
            p=self.dropout,
            training=self.training,
        )
        attended = probabilities @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        output = self.projection(attended)
        if uses_quantized_activations(self.mode):
            output = quantize_activation(
                output, self.mode, self.activation_threshold, lanes=self.population_lanes
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
            "population_lanes": config.population_lanes if mode == "population_ternary" else 1,
        }
        self.up = TernaryLinear(config.d_model, hidden_size, **linear_args)
        self.down = TernaryLinear(hidden_size, config.d_model, **linear_args)
        self.dropout = nn.Dropout(config.dropout)
        self.mode = mode
        self.activation_threshold = config.activation_threshold
        self.population_lanes = config.population_lanes if mode == "population_ternary" else 1

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.gelu(self.up(inputs), approximate="tanh")
        if uses_quantized_activations(self.mode):
            hidden = quantize_activation(
                hidden, self.mode, self.activation_threshold, lanes=self.population_lanes
            )
        output = self.down(hidden)
        output = self.dropout(output)
        if uses_quantized_activations(self.mode):
            output = quantize_activation(
                output, self.mode, self.activation_threshold, lanes=self.population_lanes
            )
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, mode: Mode) -> None:
        super().__init__()
        norm_args = {
            "mode": mode,
            "weight_threshold": config.weight_threshold,
            "activation_threshold": config.activation_threshold,
            "population_lanes": config.population_lanes if mode == "population_ternary" else 1,
        }
        self.attention_norm = TernaryRMSNorm(config.d_model, **norm_args)
        self.attention = CausalSelfAttention(config, mode)
        self.feed_forward_norm = TernaryRMSNorm(config.d_model, **norm_args)
        self.feed_forward = FeedForward(config, mode)
        self.mode = mode
        self.activation_threshold = config.activation_threshold
        self.population_lanes = config.population_lanes if mode == "population_ternary" else 1
        self.residual_scale = config.residual_scale
        projection = normalized_hadamard(config.d_model)
        self.register_buffer("activation_projection", projection, persistent=True)
        self.register_buffer(
            "coat_projection_ready",
            torch.tensor(not requires_coat_calibration(mode)),
            persistent=True,
        )

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
            return quantize_projected_activation(
                tensor,
                self.mode,
                self.activation_threshold,
                self.activation_projection,
                lanes=self.population_lanes,
            )
        return quantize_activation(
            tensor, self.mode, self.activation_threshold, lanes=self.population_lanes
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
        self.blocks = nn.ModuleList(TransformerBlock(config, mode) for _ in range(config.n_layers))
        self.final_norm = TernaryRMSNorm(
            config.d_model,
            mode=mode,
            weight_threshold=config.weight_threshold,
            activation_threshold=config.activation_threshold,
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

    def load_activation_projection(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        projection = payload["projection"] if isinstance(payload, dict) else payload
        self.set_activation_projection(projection)

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
            "aggregate": aggregate,
            "layers": per_layer,
        }

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
            },
        }
