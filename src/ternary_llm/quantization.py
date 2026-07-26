from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from ternary_llm.config import (
    ActivationEncoding,
    AttentionNormalization,
    AttentionQuantization,
    Mode,
)


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


def residual_plane_ternary_linear_reference(
    inputs: Tensor,
    weight: Tensor,
    *,
    planes: int,
    binary: bool,
    activation_threshold: float = 0.5,
    weight_threshold: float = 0.5,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Apply a linear layer as code-plane dot products with INT32 accumulators.

    This is a deliberately simple deployment reference, not an optimized
    kernel. It keeps the large matrix-multiplication operands as binary/ternary
    codes. Per-vector activation scales and per-row weight scales are applied
    only after the exact integer dot products have been accumulated.
    """
    if inputs.ndim < 2:
        raise ValueError("inputs must have at least two dimensions")
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    if inputs.shape[-1] != weight.shape[-1]:
        raise ValueError(
            "input and weight reduction dimensions must match: "
            f"{inputs.shape[-1]} != {weight.shape[-1]}"
        )

    activation_codes, activation_scales = residual_refinement_activation_codes(
        inputs,
        planes,
        binary=binary,
        threshold=activation_threshold,
        eps=eps,
    )
    weight_scale = _scale(weight, 1, eps=eps)
    weight_codes = ternary_code(
        weight.detach() / weight_scale,
        weight_threshold,
    )

    reduction = inputs.shape[-1]
    output_features = weight.shape[0]
    flattened_codes = activation_codes.reshape(-1, reduction, planes)
    plane_major_codes = flattened_codes.permute(0, 2, 1).reshape(-1, reduction)
    integer_accumulators = (
        plane_major_codes.to(torch.int32) @ weight_codes.T.to(torch.int32)
    )
    integer_accumulators = integer_accumulators.reshape(
        -1,
        planes,
        output_features,
    ).permute(0, 2, 1)

    flattened_scales = activation_scales.reshape(-1, 1, planes)
    output = (
        integer_accumulators.to(inputs.dtype) * flattened_scales
    ).sum(dim=-1)
    output = output * weight_scale.reshape(1, output_features).to(inputs.dtype)
    output = output.reshape(*inputs.shape[:-1], output_features)
    accumulator_shape = (*inputs.shape[:-1], output_features, planes)
    return output, integer_accumulators.reshape(accumulator_shape)


def _integer_sqrt_tensor(values: Tensor) -> Tensor:
    """Portable exact integer square root for a correctness reference."""
    if values.dtype != torch.int64:
        raise ValueError("integer square root expects int64 values")
    if bool(torch.any(values < 0)):
        raise ValueError("integer square root expects non-negative values")
    original_device = values.device
    roots = [math.isqrt(int(value)) for value in values.detach().cpu().reshape(-1)]
    return torch.tensor(roots, dtype=torch.int64, device=original_device).reshape(
        values.shape
    )


def integer_rms_norm_reference(
    inputs: Tensor,
    weight: Tensor,
    *,
    input_fraction_bits: int = 14,
    output_fraction_bits: int = 14,
    weight_threshold: float = 0.5,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Apply RMSNorm with integer reductions and a ternary weight operand.

    The portable implementation quantizes the incoming boundary to signed
    fixed-point integers, accumulates squares in INT64, uses an exact integer
    square root and integer division, and applies ternary normalization-weight
    codes. The small per-vector and per-weight scales are reconstructed only
    after the integer elementwise work. It is a correctness reference rather
    than a fused kernel.
    """
    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if weight.ndim != 1 or weight.shape[0] != inputs.shape[-1]:
        raise ValueError("weight must be a vector matching the input width")
    if not 1 <= input_fraction_bits <= 20:
        raise ValueError("input_fraction_bits must be between 1 and 20")
    if not 1 <= output_fraction_bits <= 20:
        raise ValueError("output_fraction_bits must be between 1 and 20")
    if eps <= 0:
        raise ValueError("eps must be positive")

    input_multiplier = 1 << input_fraction_bits
    input_codes = (inputs.detach().to(torch.float64) * input_multiplier).round()
    input_codes = input_codes.clamp(-(1 << 31), (1 << 31) - 1).to(torch.int64)
    width = inputs.shape[-1]
    sum_squares = (input_codes * input_codes).sum(dim=-1, keepdim=True)
    mean_square = (sum_squares + width // 2) // width
    epsilon_code = max(1, round(eps * input_multiplier * input_multiplier))
    rms_codes = _integer_sqrt_tensor(mean_square + epsilon_code).clamp_min(1)

    output_multiplier = 1 << output_fraction_bits
    numerator = input_codes * output_multiplier
    half_denominator = rms_codes // 2
    normalized_codes = torch.where(
        numerator >= 0,
        (numerator + half_denominator) // rms_codes,
        -((-numerator + half_denominator) // rms_codes),
    )

    weight_scale = _scale(weight, 0, eps=eps)
    weight_codes = ternary_code(
        weight.detach() / weight_scale,
        weight_threshold,
    ).to(torch.int64)
    weighted_codes = normalized_codes * weight_codes
    output = weighted_codes.to(inputs.dtype) / output_multiplier
    output = output * weight_scale.to(device=inputs.device, dtype=inputs.dtype)
    return output, normalized_codes.to(torch.int32)


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


def binary_activation_codes(
    tensor: Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Return exact {-1, +1} codes and their optimal per-vector scale."""
    codes = torch.where(
        tensor.detach() >= 0,
        torch.ones_like(tensor),
        -torch.ones_like(tensor),
    )
    scale = _scale(tensor, -1, eps=eps)
    return codes, scale


def binarize_activation_with_learned_scale(
    tensor: Tensor,
    scale: Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Binarize with a broadcastable learned scale and STE gradients."""
    positive_scale = scale.clamp_min(eps)
    codes = torch.where(
        tensor.detach() >= 0,
        torch.ones_like(tensor),
        -torch.ones_like(tensor),
    )
    dequantized = codes * positive_scale
    quantized = (
        tensor
        + (dequantized - tensor).detach()
        + (dequantized - dequantized.detach())
    )
    return quantized, codes


def ternarize_activation_with_learned_scale(
    tensor: Tensor,
    scale: Tensor,
    threshold: float = 0.5,
    *,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Ternarize with a broadcastable learned scale and STE gradients.

    Codes are selected from detached values and scales. The returned activation
    passes an identity gradient to the shadow activation and a code-weighted
    gradient to the scale, while its forward values are exactly code * scale.
    """
    positive_scale = scale.clamp_min(eps)
    codes = ternary_code(
        tensor.detach() / positive_scale.detach(),
        threshold,
    )
    dequantized = codes * positive_scale
    quantized = (
        tensor
        + (dequantized - tensor).detach()
        + (dequantized - dequantized.detach())
    )
    return quantized, codes


def ternary_qk_attention_reference(
    query: Tensor,
    key: Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Compute scaled Q·K scores from ternary codes and INT32 accumulators.

    Query and key may have different sequence lengths, but their batch/head
    prefix dimensions and head width must match. Per-vector scales are applied
    after the integer dot product, so both large matrix operands remain exact
    ternary codes.
    """
    if query.ndim < 2 or key.ndim < 2:
        raise ValueError("query and key must have at least two dimensions")
    if query.shape[:-2] != key.shape[:-2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "query and key must share prefix dimensions and head width"
        )
    query_codes, query_scale = ternary_activation_codes(
        query,
        threshold,
        eps=eps,
    )
    key_codes, key_scale = ternary_activation_codes(
        key,
        threshold,
        eps=eps,
    )
    accumulators = (
        query_codes.to(torch.int32)
        @ key_codes.transpose(-2, -1).to(torch.int32)
    )
    scores = accumulators.to(query.dtype)
    scores = scores * query_scale.to(query.dtype)
    scores = scores * key_scale.transpose(-2, -1).to(query.dtype)
    scores = scores / query.shape[-1] ** 0.5
    return scores, accumulators


def binary_qk_attention_reference(
    query: Tensor,
    key: Tensor,
    *,
    query_scale: Tensor | None = None,
    key_scale: Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Compute binary Q·K scores with exact INT32 accumulation.

    Optional scales allow the learned per-head deployment path to move both
    positive scale factors outside the binary dot product. Without explicit
    scales, the function uses magnitude-optimal per-vector references.
    """
    if query.ndim < 2 or key.ndim < 2:
        raise ValueError("query and key must have at least two dimensions")
    if query.shape[:-2] != key.shape[:-2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "query and key must share prefix dimensions and head width"
        )
    query_codes, dynamic_query_scale = binary_activation_codes(
        query,
        eps=eps,
    )
    key_codes, dynamic_key_scale = binary_activation_codes(
        key,
        eps=eps,
    )
    resolved_query_scale = (
        dynamic_query_scale
        if query_scale is None
        else query_scale.detach().clamp_min(eps)
    )
    resolved_key_scale = (
        dynamic_key_scale
        if key_scale is None
        else key_scale.detach().clamp_min(eps)
    )
    try:
        torch.broadcast_shapes(query.shape, resolved_query_scale.shape)
        torch.broadcast_shapes(key.shape, resolved_key_scale.shape)
    except RuntimeError as error:
        raise ValueError("Q/K scales must be broadcastable to their operands") from error
    accumulators = (
        query_codes.to(torch.int32)
        @ key_codes.transpose(-2, -1).to(torch.int32)
    )
    scores = accumulators.to(query.dtype)
    scores = scores * resolved_query_scale.to(query.dtype)
    scores = scores * resolved_key_scale.transpose(-2, -1).to(query.dtype)
    scores = scores / query.shape[-1] ** 0.5
    return scores, accumulators


def int2_route_ternary_value_reference(
    probability_codes: Tensor,
    values: Tensor,
    value_scale: Tensor,
    *,
    empty_route_code: Tensor | int = 0,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Compute Route·V from two binary route planes and ternary V codes.

    This reference requires a value scale broadcastable across the key-token
    dimension, as in the learned head-shared scale experiment. Two-bit route
    codes are decomposed into low/high binary planes, so both large matmuls use
    binary-by-ternary operands and INT32 accumulators.
    """
    if probability_codes.ndim < 2 or values.ndim < 2:
        raise ValueError("probability codes and values need at least two dimensions")
    if probability_codes.shape[:-2] != values.shape[:-2]:
        raise ValueError("probability codes and values must share prefix dimensions")
    if probability_codes.shape[-1] != values.shape[-2]:
        raise ValueError("route key length must equal the value sequence length")
    detached_codes = probability_codes.detach()
    if not torch.equal(detached_codes, detached_codes.round()):
        raise ValueError("probability codes must be integers")
    if detached_codes.numel() and (
        detached_codes.min().item() < 0 or detached_codes.max().item() > 3
    ):
        raise ValueError("probability codes must be in [0, 3]")

    positive_scale = value_scale.detach().clamp_min(eps)
    value_codes = ternary_code(
        values.detach() / positive_scale,
        threshold,
    ).to(torch.int32)
    route_codes = detached_codes.to(torch.int32)
    low_plane = torch.bitwise_and(route_codes, 1)
    high_plane = torch.bitwise_and(torch.bitwise_right_shift(route_codes, 1), 1)
    low_accumulator = low_plane @ value_codes
    high_accumulator = high_plane @ value_codes
    accumulators = torch.stack((low_accumulator, high_accumulator), dim=-1)
    combined = low_accumulator + torch.bitwise_left_shift(high_accumulator, 1)
    empty_code = torch.as_tensor(
        empty_route_code,
        device=route_codes.device,
        dtype=torch.int32,
    )
    if empty_code.numel() and (
        empty_code.min().item() < 0 or empty_code.max().item() > 3
    ):
        raise ValueError("empty route code must be in [0, 3]")
    denominator = (
        route_codes.sum(dim=-1, keepdim=True) + empty_code
    ).clamp_min(1)
    output = combined.to(values.dtype) / denominator.to(values.dtype)
    output = output * positive_scale.to(values.dtype)
    return output, accumulators


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
    normalization: AttentionNormalization = "softmax",
    clip: float,
    threshold: float,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Return attention probabilities plus optional score/probability codes."""
    if normalization == "softmax1":
        # A virtual key with a fixed zero score implements
        # exp(score_i) / (1 + sum_j exp(score_j)). It is included before score
        # shifting and route quantization, but removed before Route·V.
        scores = torch.cat((scores, torch.zeros_like(scores[..., :1])), dim=-1)
        valid = torch.cat((valid, torch.ones_like(valid[..., :1])), dim=-1)
    elif normalization != "softmax":
        raise ValueError(f"unsupported attention normalization: {normalization}")
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
    if normalization == "softmax1":
        probabilities = probabilities[..., :-1]
        if score_codes is not None:
            score_codes = score_codes[..., :-1]
        if probability_codes is not None:
            probability_codes = probability_codes[..., :-1]
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
