"""Native ternary QAT primitives with changeable {-1, 0, +1} codes.

Unlike the scale-only recovery pilot, this module keeps a high-precision shadow
weight while training and uses an exact ternary weight in every forward pass.
The straight-through estimator (STE) lets the shadow weight cross ternary
decision boundaries.  Exported artifacts contain only codes and scales; shadow
weights are a training implementation detail and are never part of inference.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from ternary_llm.minicpmo_qat import _inverse_softplus, groupwise_ternary_codebook


class DynamicTernarySTE(nn.Module):
    """Parametrize a weight with dynamic ternary codes and learned group scales.

    ``original`` remains the trainable shadow weight.  The numerical forward
    value is always ``code * scale`` while its gradient with respect to the
    shadow weight is the identity STE.  Codes are recalculated on every forward
    pass, so training can repair a bad post-training rounding decision.
    """

    def __init__(
        self,
        *,
        weight: Tensor,
        group_size: int = 512,
        threshold: float = 0.5,
        lloyd_iterations: int = 8,
        minimum_scale: float = 1e-6,
        train_scales: bool = True,
    ) -> None:
        super().__init__()
        if group_size < 1:
            raise ValueError("group_size must be positive")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")

        _codes, scales = groupwise_ternary_codebook(
            weight,
            group_size=group_size,
            threshold=threshold,
            lloyd_iterations=lloyd_iterations,
        )
        initial = scales.detach().to(torch.float32).clamp_min(minimum_scale * 2)
        self.raw_scales = nn.Parameter(
            _inverse_softplus(initial - minimum_scale),
            requires_grad=train_scales,
        )
        self.shape = tuple(weight.shape)
        self.group_size = group_size
        self.threshold = threshold
        self.minimum_scale = minimum_scale

    @property
    def scales(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_scales) + self.minimum_scale

    def _grouped_shadow(self, original: Tensor) -> Tensor:
        flat = original.reshape(-1)
        padding = (-flat.numel()) % self.group_size
        if padding:
            flat = torch.nn.functional.pad(flat, (0, padding))
        return flat.reshape(-1, self.group_size)

    def codes(self, original: Tensor) -> Tensor:
        grouped = self._grouped_shadow(original.detach())
        scales = self.scales.detach().to(dtype=grouped.dtype)
        normalized = grouped / scales
        return (normalized > self.threshold).to(torch.int8) - (
            normalized < -self.threshold
        ).to(torch.int8)

    def forward(self, original: Tensor) -> Tensor:
        codes = self.codes(original).to(dtype=original.dtype)
        scales = self.scales.to(dtype=original.dtype)
        quantized = (codes * scales).reshape(-1)[: math.prod(self.shape)]
        quantized = quantized.reshape(self.shape).to(dtype=original.dtype)

        # Forward: exactly quantized. Backward: identity for the shadow weight
        # plus the real gradient for learned group scales.
        return (
            original
            + (quantized - original).detach()
            + (quantized - quantized.detach())
        )

    def artifact_payload(self, original: Tensor) -> dict[str, Any]:
        return {
            "codes": self.codes(original).cpu(),
            "scales": self.scales.detach().to(torch.float16).cpu(),
            "shape": list(self.shape),
            "group_size": self.group_size,
            "threshold": self.threshold,
            "code_values": [-1, 0, 1],
            "shadow_weights_exported": False,
        }


def attach_dynamic_ternary_qat(
    module: nn.Module,
    parameter_name: str,
    *,
    group_size: int = 512,
    threshold: float = 0.5,
    lloyd_iterations: int = 8,
    train_scales: bool = True,
) -> DynamicTernarySTE:
    """Attach dynamic-code ternary QAT to one parameter."""

    parameter = getattr(module, parameter_name)
    parameter.requires_grad_(True)
    parametrization = DynamicTernarySTE(
        weight=parameter,
        group_size=group_size,
        threshold=threshold,
        lloyd_iterations=lloyd_iterations,
        train_scales=train_scales,
    ).to(parameter.device)
    parametrize.register_parametrization(module, parameter_name, parametrization)
    return parametrization
