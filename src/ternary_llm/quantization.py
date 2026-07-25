from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ternary_llm.config import ActivationEncoding, AttentionQuantization, Mode


def uses_ternary_weights(mode: Mode) -> bool:
    return mode in {
        "ternary_weights",
        "ternary_forward",
        "population_ternary",
        "hadamard_a4",
        "coat_a4",
        "hadamard_ternary",
        "coat_ternary",
        "coat_progressive",
        "hadamard_progressive",
    }


def uses_ternary_activations(mode: Mode) -> bool:
    return mode in {
        "ternary_activations",
        "ternary_forward",
        "population_ternary",
        "hadamard_ternary",
        "coat_ternary",
        "coat_progressive",
        "hadamard_progressive",
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
        "coat_progressive",
        "hadamard_progressive",
    }


def requires_coat_calibration(mode: Mode) -> bool:
    return mode in {"coat_a4", "coat_ternary", "coat_progressive"}


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


def quantize_activation_levels(
    tensor: Tensor,
    levels: int,
    *,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> Tensor:
    """Quantize per vector to a symmetric odd alphabet with magnitude alignment.

    The integer code alphabet is ``[-qmax, ..., 0, ..., qmax]``. The grid
    resolution grows with ``sqrt(qmax)``: high-level stages obtain a finer step
    and wider range, while reducing levels smoothly converges to the ordinary
    ternary threshold. The dequantization scale is then aligned so the
    quantized vector preserves the original mean absolute magnitude. At three
    levels the codes are exactly ternary. The forward path is quantized and the
    backward path is an STE.
    """
    if levels < 3 or levels % 2 == 0:
        raise ValueError("levels must be an odd integer of at least 3")
    codes, aligned_scale = progressive_activation_codes(
        tensor,
        levels,
        threshold=threshold,
        eps=eps,
    )
    dequantized = codes * aligned_scale
    return tensor + (dequantized - tensor).detach()


def residual_refinement_activation_codes(
    tensor: Tensor,
    planes: int,
    *,
    binary: bool,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Greedily encode an activation as scaled binary or ternary residual planes.

    Each plane is fitted to the residual left by earlier planes. Binary planes
    use exactly one bit per scalar and are a subset of ternary arithmetic.
    Ternary planes add an explicit zero code. The returned scales are
    least-squares-optimal per vector for the selected codes.
    """
    if planes < 1:
        raise ValueError("planes must be positive")
    residual = tensor.detach()
    plane_codes = []
    plane_scales = []
    for _ in range(planes):
        initial_scale = _scale(residual, -1, eps=eps)
        normalized = residual / initial_scale
        if binary:
            codes = torch.where(
                normalized >= 0,
                torch.ones_like(normalized),
                -torch.ones_like(normalized),
            )
        else:
            codes = ternary_code(normalized, threshold)
        denominator = codes.square().sum(dim=-1, keepdim=True).clamp_min(1.0)
        scale = (residual * codes).sum(dim=-1, keepdim=True) / denominator
        scale = scale.abs().clamp_min(eps)
        residual = residual - codes * scale
        plane_codes.append(codes)
        plane_scales.append(scale)
    return torch.stack(plane_codes, dim=-1), torch.stack(plane_scales, dim=-1)


def quantize_activation_residual_planes(
    tensor: Tensor,
    planes: int,
    *,
    binary: bool,
    threshold: float = 0.5,
) -> Tensor:
    """Fake-quantize an activation to a sum of scaled low-bit planes."""
    codes, scales = residual_refinement_activation_codes(
        tensor,
        planes,
        binary=binary,
        threshold=threshold,
    )
    dequantized = (codes * scales).sum(dim=-1)
    return tensor + (dequantized - tensor).detach()


def progressive_activation_codes(
    tensor: Tensor,
    levels: int,
    *,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Return integer codes and a magnitude-aligned per-vector scale."""
    if levels < 3 or levels % 2 == 0:
        raise ValueError("levels must be an odd integer of at least 3")
    qmax = (levels - 1) // 2
    magnitude = _scale(tensor, -1, eps=eps)
    resolution = qmax**0.5
    normalized = tensor.detach() / magnitude * resolution
    if levels == 3:
        codes = ternary_code(normalized, threshold)
    else:
        codes = normalized.round().clamp(-qmax, qmax)
    code_magnitude = codes.abs().mean(dim=-1, keepdim=True)
    aligned_scale = magnitude / code_magnitude.clamp_min(eps)
    return codes, aligned_scale


def ternary_activation_codes(
    tensor: Tensor,
    threshold: float = 0.5,
    *,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Return per-vector ternary codes and their detached dynamic scales."""
    scale = _scale(tensor, -1, eps=eps)
    return ternary_code(tensor.detach() / scale, threshold), scale


def quantize_activation_a4(tensor: Tensor, eps: float = 1e-5) -> Tensor:
    """Per-vector asymmetric fake quantization to 16 activation levels."""
    minimum = tensor.detach().amin(dim=-1, keepdim=True)
    maximum = tensor.detach().amax(dim=-1, keepdim=True)
    scale = ((maximum - minimum) / 15.0).clamp_min(eps)
    codes = ((tensor.detach() - minimum) / scale).round().clamp(0, 15)
    dequantized = codes * scale + minimum
    return tensor + (dequantized - tensor).detach()


def quantize_attention_scores_int2(
    scores: Tensor,
    valid: Tensor,
    *,
    clip: float,
) -> tuple[Tensor, Tensor]:
    """Quantize max-shifted valid softmax inputs to four codes {-3,-2,-1,0}.

    Invalid causal positions remain negative infinity. The returned integer
    codes are useful for measuring code utilization and emulating a four-entry
    exponential lookup table.
    """
    if clip <= 0:
        raise ValueError("clip must be positive")
    masked = scores.masked_fill(~valid, -torch.inf)
    row_max = masked.amax(dim=-1, keepdim=True)
    shifted = (masked - row_max).clamp(min=-clip, max=0.0)
    step = clip / 3.0
    codes = (shifted.detach() / step).round().clamp(-3, 0)
    dequantized = codes * step
    quantized = shifted + (dequantized - shifted).detach()
    quantized = quantized.masked_fill(~valid, -torch.inf)
    return quantized, codes.masked_fill(~valid, 0)


def quantize_attention_probabilities_int2(
    probabilities: Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Quantize each attention row to four non-negative levels and renormalize."""
    scale = (probabilities.detach().amax(dim=-1, keepdim=True) / 3.0).clamp_min(eps)
    codes = (probabilities.detach() / scale).round().clamp(0, 3)
    dequantized = codes * scale
    normalized = dequantized / dequantized.sum(dim=-1, keepdim=True).clamp_min(eps)
    quantized = probabilities + (normalized - probabilities).detach()
    return quantized, codes


def quantize_attention_probabilities_binary(
    probabilities: Tensor,
    *,
    threshold: float,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Binarize attention relative to each row maximum and renormalize.

    The row maximum always survives for thresholds in (0, 1], so every row has
    at least one active route.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    cutoff = probabilities.detach().amax(dim=-1, keepdim=True) * threshold
    codes = (probabilities.detach() >= cutoff).to(probabilities.dtype)
    normalized = codes / codes.sum(dim=-1, keepdim=True).clamp_min(eps)
    quantized = probabilities + (normalized - probabilities).detach()
    return quantized, codes


def integer_softmax_from_int2_codes(
    scores: Tensor,
    codes: Tensor,
    valid: Tensor,
    *,
    clip: float,
    fraction_bits: int = 15,
) -> tuple[Tensor, Tensor]:
    """Emulate a four-entry integer exponential LUT and integer row sum.

    The forward values are produced from Q-format integer numerators and an
    integer denominator. The backward path follows ordinary softmax, which is
    the straight-through surrogate used during QAT.
    """
    if fraction_bits < 1 or fraction_bits > 30:
        raise ValueError("fraction_bits must be between 1 and 30")
    step = clip / 3.0
    codebook = torch.arange(-3, 1, device=scores.device, dtype=torch.float32)
    multiplier = 1 << fraction_bits
    lookup = (torch.exp(codebook * step) * multiplier).round().clamp_min(1)
    indices = (codes.to(torch.long) + 3).clamp(0, 3)
    integer_numerators = lookup[indices].to(scores.dtype)
    integer_numerators = integer_numerators.masked_fill(~valid, 0)
    integer_denominator = integer_numerators.sum(dim=-1, keepdim=True).clamp_min(1)
    fixed = integer_numerators / integer_denominator
    surrogate = torch.softmax(scores, dim=-1)
    probabilities = surrogate + (fixed - surrogate).detach()
    return probabilities, integer_numerators


def quantize_attention(
    scores: Tensor,
    valid: Tensor,
    *,
    scheme: AttentionQuantization,
    clip: float,
    threshold: float,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Return attention probabilities plus optional score/probability codes."""
    score_codes = None
    score_schemes = {
        "score_int2",
        "score_int2_lut",
        "score_prob_int2",
        "score_lut_prob_int2",
        "score_lut_prob_binary",
    }
    if scheme in score_schemes:
        scores, score_codes = quantize_attention_scores_int2(scores, valid, clip=clip)
    else:
        scores = scores.masked_fill(~valid, -torch.inf)
    if scheme in {"score_int2_lut", "score_lut_prob_int2", "score_lut_prob_binary"}:
        if score_codes is None:
            raise AssertionError("integer softmax requires score codes")
        probabilities, _ = integer_softmax_from_int2_codes(
            scores,
            score_codes,
            valid,
            clip=clip,
        )
    else:
        probabilities = torch.softmax(scores, dim=-1)
    probability_codes = None
    if scheme in {"prob_int2", "score_prob_int2", "score_lut_prob_int2"}:
        probabilities, probability_codes = quantize_attention_probabilities_int2(
            probabilities
        )
    elif scheme in {"prob_binary", "score_lut_prob_binary"}:
        probabilities, probability_codes = quantize_attention_probabilities_binary(
            probabilities,
            threshold=threshold,
        )
    return probabilities, score_codes, probability_codes


def quantize_activation(
    tensor: Tensor,
    mode: Mode,
    threshold: float,
    *,
    lanes: int,
    levels: int = 19,
    encoding: ActivationEncoding = "uniform",
    planes: int = 1,
) -> Tensor:
    if mode in {"coat_progressive", "hadamard_progressive"}:
        if encoding == "residual_binary":
            return quantize_activation_residual_planes(
                tensor,
                planes,
                binary=True,
                threshold=threshold,
            )
        if encoding == "residual_ternary":
            return quantize_activation_residual_planes(
                tensor,
                planes,
                binary=False,
                threshold=threshold,
            )
        return quantize_activation_levels(tensor, levels, threshold=threshold)
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
    levels: int = 19,
    encoding: ActivationEncoding = "uniform",
    planes: int = 1,
) -> Tensor:
    """Encode in an orthogonal basis, then decode for the reference kernels."""
    if not uses_activation_projection(mode):
        return quantize_activation(
            tensor,
            mode,
            threshold,
            lanes=lanes,
            levels=levels,
            encoding=encoding,
            planes=planes,
        )
    q = projection.to(device=tensor.device, dtype=tensor.dtype)
    rotated = tensor @ q
    quantized = quantize_activation(
        rotated,
        mode,
        threshold,
        lanes=lanes,
        levels=levels,
        encoding=encoding,
        planes=planes,
    )
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
