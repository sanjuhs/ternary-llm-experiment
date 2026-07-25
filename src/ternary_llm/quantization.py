from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ternary_llm.config import Mode


def uses_ternary_weights(mode: Mode) -> bool:
    return mode in {
        "ternary_weights",
        "ternary_forward",
        "population_ternary",
        "hadamard_a4",
        "coat_a4",
        "hadamard_ternary",
        "coat_ternary",
    }


def uses_ternary_activations(mode: Mode) -> bool:
    return mode in {
        "ternary_activations",
        "ternary_forward",
        "population_ternary",
        "hadamard_ternary",
        "coat_ternary",
    }


def uses_a4_activations(mode: Mode) -> bool:
    return mode in {"hadamard_a4", "coat_a4"}


def uses_quantized_activations(mode: Mode) -> bool:
    return uses_ternary_activations(mode) or uses_a4_activations(mode)


def uses_activation_projection(mode: Mode) -> bool:
    return mode in {
        "hadamard_a4",
        "coat_a4",
        "hadamard_ternary",
        "coat_ternary",
    }


def requires_coat_calibration(mode: Mode) -> bool:
    return mode in {"coat_a4", "coat_ternary"}


def _scale(
    tensor: Tensor,
    dimensions: int | Sequence[int],
    *,
    eps: float,
) -> Tensor:
    return tensor.detach().abs().mean(dim=dimensions, keepdim=True).clamp_min(eps)


def ternary_code(normalized: Tensor, threshold: float = 0.5) -> Tensor:
    """Return exact codes in {-1, 0, +1} for a normalized tensor."""
    positive = normalized > threshold
    negative = normalized < -threshold
    return positive.to(normalized.dtype) - negative.to(normalized.dtype)


def ternarize(
    tensor: Tensor,
    *,
    dimensions: int | Sequence[int],
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> Tensor:
    """Scaled ternary fake quantization with a straight-through gradient."""
    scale = _scale(tensor, dimensions, eps=eps)
    code = ternary_code(tensor / scale, threshold)
    dequantized = code * scale
    return tensor + (dequantized - tensor).detach()


def population_ternary_codes(
    tensor: Tensor,
    *,
    lanes: int,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Encode each scalar as deterministic thermometer-style ternary lanes.

    The returned code tensor has one additional final dimension. Its values are
    exactly -1, 0, or +1. The detached per-token scale is returned separately.
    """
    if lanes < 1:
        raise ValueError("lanes must be positive")
    scale = _scale(tensor, -1, eps=eps)
    normalized = tensor.detach() / scale
    lane_thresholds = (
        (torch.arange(lanes, device=tensor.device, dtype=tensor.dtype) + 0.5)
        * (2.0 * threshold / lanes)
    )
    active = normalized.abs().unsqueeze(-1) > lane_thresholds
    signs = normalized.sign().unsqueeze(-1)
    codes = active.to(tensor.dtype) * signs
    return codes, scale


def ternarize_activation(
    tensor: Tensor,
    threshold: float = 0.5,
    *,
    lanes: int = 1,
) -> Tensor:
    """Quantize each token/vector, optionally using population-coded lanes."""
    if lanes == 1:
        return ternarize(tensor, dimensions=-1, threshold=threshold)
    codes, scale = population_ternary_codes(tensor, lanes=lanes, threshold=threshold)
    dequantized = codes.mean(dim=-1) * scale
    return tensor + (dequantized - tensor).detach()


def quantize_activation_a4(tensor: Tensor, eps: float = 1e-5) -> Tensor:
    """Per-vector asymmetric fake quantization to 16 activation levels."""
    minimum = tensor.detach().amin(dim=-1, keepdim=True)
    maximum = tensor.detach().amax(dim=-1, keepdim=True)
    scale = ((maximum - minimum) / 15.0).clamp_min(eps)
    codes = ((tensor.detach() - minimum) / scale).round().clamp(0, 15)
    dequantized = codes * scale + minimum
    return tensor + (dequantized - tensor).detach()


def quantize_activation(tensor: Tensor, mode: Mode, threshold: float, *, lanes: int) -> Tensor:
    if uses_a4_activations(mode):
        return quantize_activation_a4(tensor)
    if uses_ternary_activations(mode):
        return ternarize_activation(tensor, threshold, lanes=lanes)
    return tensor


def quantize_projected_activation(
    tensor: Tensor,
    mode: Mode,
    threshold: float,
    projection: Tensor,
    *,
    lanes: int,
) -> Tensor:
    """Encode in an orthogonal basis, then decode for the reference kernels."""
    if not uses_activation_projection(mode):
        return quantize_activation(tensor, mode, threshold, lanes=lanes)
    q = projection.to(device=tensor.device, dtype=tensor.dtype)
    rotated = tensor @ q
    quantized = quantize_activation(rotated, mode, threshold, lanes=lanes)
    return quantized @ q.T


def ternarize_weight(tensor: Tensor, threshold: float = 0.5) -> Tensor:
    """Quantize per output row (or globally for one-dimensional parameters)."""
    dimensions: int | tuple[int, ...] = 0 if tensor.ndim <= 1 else tuple(range(1, tensor.ndim))
    return ternarize(tensor, dimensions=dimensions, threshold=threshold)


def code_histogram(tensor: Tensor, threshold: float = 0.5) -> dict[str, int]:
    dimensions: int | tuple[int, ...] = 0 if tensor.ndim <= 1 else tuple(range(1, tensor.ndim))
    scale = _scale(tensor, dimensions, eps=1e-5)
    codes = ternary_code(tensor.detach() / scale, threshold)
    return {
        "-1": int((codes == -1).sum().item()),
        "0": int((codes == 0).sum().item()),
        "+1": int((codes == 1).sum().item()),
    }
