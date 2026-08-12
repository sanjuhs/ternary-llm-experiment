import pytest
import torch

from ternary_llm.quantization import (
    binarize_activation_with_learned_scale,
    binary_activation_codes,
    binary_qk_attention_reference,
    int2_route_ternary_value_reference,
    integer_rms_norm_reference,
    integer_softmax_from_int2_codes,
    population_ternary_codes,
    progressive_activation_codes,
    quantize_activation_a4,
    quantize_activation_levels,
    quantize_activation_residual_planes,
    quantize_attention,
    quantize_attention_probabilities_binary,
    quantize_attention_probabilities_int2,
    quantize_attention_scores_int2,
    residual_plane_ternary_linear_reference,
    residual_refinement_activation_codes,
    ternarize,
    ternarize_activation,
    ternarize_activation_with_learned_scale,
    ternarize_weight,
    ternary_activation_codes,
    ternary_code,
    ternary_qk_attention_reference,
)


def test_ternary_code_has_exact_three_value_alphabet() -> None:
    values = torch.tensor([-2.0, -0.6, -0.4, 0.0, 0.4, 0.6, 2.0])
    codes = ternary_code(values, threshold=0.5)
    assert codes.tolist() == [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_straight_through_estimator_passes_gradient() -> None:
    values = torch.tensor([[0.1, -0.2, 0.7]], requires_grad=True)
    output = ternarize(values, dimensions=-1).sum()
    output.backward()
    assert values.grad is not None
    assert torch.equal(values.grad, torch.ones_like(values))


def test_population_codes_are_ternary_and_add_resolution() -> None:
    values = torch.tensor([[-0.8, -0.3, 0.0, 0.3, 0.8]])
    codes, _ = population_ternary_codes(values, lanes=4)

    assert codes.shape == (1, 5, 4)
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    decoded = ternarize_activation(values, lanes=4)
    assert decoded.unique().numel() > ternarize_activation(values, lanes=1).unique().numel()


def test_population_activation_uses_straight_through_gradient() -> None:
    values = torch.tensor([[0.1, -0.2, 0.7]], requires_grad=True)
    ternarize_activation(values, lanes=4).sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_a4_activation_uses_straight_through_gradient() -> None:
    values = torch.tensor([[0.1, -0.2, 0.7]], requires_grad=True)
    quantize_activation_a4(values).sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_progressive_activation_uses_requested_alphabet_and_alignment() -> None:
    values = torch.tensor([[-3.0, -1.1, -0.2, 0.0, 0.4, 1.2, 3.5]], requires_grad=True)
    quantized = quantize_activation_levels(values, 7)
    input_magnitude = values.detach().abs().mean(dim=-1)
    output_magnitude = quantized.detach().abs().mean(dim=-1)

    assert torch.allclose(output_magnitude, input_magnitude, atol=1e-6)
    quantized.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_progressive_three_level_endpoint_has_ternary_codes() -> None:
    values = torch.tensor([[-2.0, -0.6, -0.1, 0.0, 0.1, 0.6, 2.0]])
    magnitude = values.abs().mean(dim=-1, keepdim=True)
    quantized = quantize_activation_levels(values, 3)
    nonzero = quantized[quantized != 0]

    assert quantized[0, 3].item() == 0.0
    assert nonzero.abs().unique().numel() == 1
    assert torch.allclose(quantized.abs().mean(dim=-1), magnitude.squeeze(-1))

    codes, _ = progressive_activation_codes(values, 3)
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_progressive_high_level_stage_has_finer_reconstruction() -> None:
    values = torch.linspace(-3.0, 3.0, 257).unsqueeze(0)
    level_19 = quantize_activation_levels(values, 19)
    level_7 = quantize_activation_levels(values, 7)

    error_19 = (level_19 - values).square().mean()
    error_7 = (level_7 - values).square().mean()
    assert error_19 < error_7


def test_binary_residual_refinement_uses_exact_one_bit_planes() -> None:
    values = torch.tensor([[-3.0, -1.1, -0.2, 0.0, 0.4, 1.2, 3.5]])
    one_code, one_scale = residual_refinement_activation_codes(
        values,
        1,
        binary=True,
    )
    two_codes, two_scales = residual_refinement_activation_codes(
        values,
        2,
        binary=True,
    )

    assert set(two_codes.unique().tolist()) == {-1.0, 1.0}
    assert two_codes.shape == (*values.shape, 2)
    one_error = (values - (one_code * one_scale).sum(dim=-1)).square().mean()
    two_error = (values - (two_codes * two_scales).sum(dim=-1)).square().mean()
    assert two_error < one_error


def test_ternary_residual_refinement_uses_sparse_planes_and_ste() -> None:
    values = torch.tensor(
        [[-3.0, -1.1, -0.2, 0.0, 0.4, 1.2, 3.5]],
        requires_grad=True,
    )
    codes, _ = residual_refinement_activation_codes(values, 2, binary=False)
    quantized = quantize_activation_residual_planes(values, 2, binary=False)

    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert 0.0 in codes.unique().tolist()
    quantized.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


@pytest.mark.parametrize("binary", [True, False])
def test_residual_plane_linear_matches_reconstructed_reference(binary: bool) -> None:
    torch.manual_seed(7)
    inputs = torch.randn(2, 3, 8)
    weight = torch.randn(5, 8)
    planes = 2
    activation_codes, activation_scales = residual_refinement_activation_codes(
        inputs,
        planes,
        binary=binary,
    )
    weight_scale = weight.detach().abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
    weight_codes = ternary_code(weight.detach() / weight_scale)
    reconstructed_inputs = (activation_codes * activation_scales).sum(dim=-1)
    reconstructed_weight = weight_codes * weight_scale

    output, accumulators = residual_plane_ternary_linear_reference(
        inputs,
        weight,
        planes=planes,
        binary=binary,
    )
    expected = torch.nn.functional.linear(reconstructed_inputs, reconstructed_weight)

    assert accumulators.dtype == torch.int32
    assert accumulators.shape == (2, 3, 5, planes)
    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)


def test_integer_rms_norm_matches_ternary_weight_reference() -> None:
    generator = torch.Generator().manual_seed(41)
    inputs = torch.randn(2, 3, 64, generator=generator)
    weight = torch.randn(64, generator=generator)

    actual, normalized_codes = integer_rms_norm_reference(
        inputs,
        weight,
        input_fraction_bits=16,
        output_fraction_bits=16,
    )
    expected = inputs * torch.rsqrt(
        inputs.square().mean(dim=-1, keepdim=True) + 1e-5
    )
    expected = expected * ternarize_weight(weight)

    assert normalized_codes.dtype == torch.int32
    assert torch.allclose(actual, expected, atol=2e-4, rtol=2e-4)


def test_integer_rms_norm_rejects_mismatched_weight() -> None:
    with pytest.raises(ValueError, match="matching the input width"):
        integer_rms_norm_reference(torch.ones(2, 4), torch.ones(3))


def test_integer_rms_norm_integer_sqrt_is_exact_for_large_codes() -> None:
    inputs = torch.tensor(
        [
            [0.0, 1.0, -1.0, 0.5],
            [1024.0, -1024.0, 511.5, -511.5],
        ]
    )
    weight = torch.ones(4)

    actual, codes = integer_rms_norm_reference(inputs, weight)

    assert torch.isfinite(actual).all()
    assert codes.dtype == torch.int32
    assert torch.equal(codes[0], torch.tensor([0, 21845, -21845, 10923]))


def test_integer_rms_norm_clamps_before_int64_square_overflow() -> None:
    inputs = torch.full((2, 1024), 1e30)
    weight = torch.ones(1024)

    actual, codes = integer_rms_norm_reference(inputs, weight)

    assert torch.isfinite(actual).all()
    assert (codes > 0).all()


def test_int2_attention_scores_use_four_codes_and_preserve_mask() -> None:
    scores = torch.tensor([[[[1.0, 0.0, -2.0], [2.0, 1.0, 0.0], [0.0, -1.0, -4.0]]]])
    valid = torch.ones(3, 3, dtype=torch.bool).tril()
    quantized, codes = quantize_attention_scores_int2(scores, valid, clip=6.0)

    assert set(codes[valid.view(1, 1, 3, 3)].tolist()) <= {-3.0, -2.0, -1.0, 0.0}
    assert torch.isneginf(quantized[0, 0, 0, 1])


def test_int2_attention_probabilities_use_four_codes_and_sum_to_one() -> None:
    probabilities = torch.tensor([[[[0.05, 0.15, 0.30, 0.50]]]], requires_grad=True)
    quantized, codes = quantize_attention_probabilities_int2(probabilities)

    assert set(codes.unique().tolist()) <= {0.0, 1.0, 2.0, 3.0}
    assert torch.allclose(quantized.sum(dim=-1), torch.ones(1, 1, 1))
    quantized.sum().backward()
    assert probabilities.grad is not None


def test_binary_attention_keeps_at_least_the_maximum_route() -> None:
    probabilities = torch.tensor([[[[0.1, 0.2, 0.3, 0.4]]]])
    quantized, codes = quantize_attention_probabilities_binary(
        probabilities,
        threshold=1.0,
    )

    assert codes.sum().item() == 1
    assert quantized.argmax(dim=-1).item() == 3
    assert torch.allclose(quantized.sum(dim=-1), torch.ones(1, 1, 1))


def test_integer_softmax_uses_integer_lut_and_preserves_gradients() -> None:
    scores = torch.tensor([[[[0.0, -2.0, -4.0, -6.0]]]], requires_grad=True)
    codes = torch.tensor([[[[0.0, -1.0, -2.0, -3.0]]]])
    valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    probabilities, numerators = integer_softmax_from_int2_codes(
        scores,
        codes,
        valid,
        clip=6.0,
    )

    assert torch.all(numerators == numerators.round())
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1, 1, 1))
    probabilities[..., 0].sum().backward()
    assert scores.grad is not None


def test_softmax1_virtual_route_reduces_attention_mass_and_keeps_gradients() -> None:
    scores = torch.zeros((1, 1, 2, 2), requires_grad=True)
    valid = torch.ones(2, 2, dtype=torch.bool).tril()
    probabilities, score_codes, route_codes = quantize_attention(
        scores,
        valid,
        scheme="score_lut_prob_int2",
        normalization="softmax1",
        clip=2.0,
        threshold=0.5,
    )

    assert probabilities.shape == scores.shape
    assert score_codes is not None and score_codes.shape == scores.shape
    assert route_codes is not None and route_codes.shape == scores.shape
    assert torch.all(probabilities.sum(dim=-1) < 1.0)
    probabilities.sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


@pytest.mark.parametrize(
    ("clip", "expected_codes"),
    [
        (2.0, {0.0, 1.0, 2.0, 3.0}),
        (3.0, {0.0, 1.0, 3.0}),
    ],
)
def test_integer_lut_clip_controls_probability_code_utilization(
    clip: float,
    expected_codes: set[float],
) -> None:
    score_codes = torch.tensor([[[[-3.0, -2.0, -1.0, 0.0]]]])
    scores = score_codes * (clip / 3.0)
    valid = torch.ones_like(score_codes, dtype=torch.bool)
    probabilities, _ = integer_softmax_from_int2_codes(
        scores,
        score_codes,
        valid,
        clip=clip,
    )
    _, probability_codes = quantize_attention_probabilities_int2(probabilities)

    assert set(probability_codes.unique().tolist()) == expected_codes


def test_ternary_qk_reference_matches_reconstructed_attention_scores() -> None:
    torch.manual_seed(13)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 5, 8)
    query_codes, query_scale = ternary_activation_codes(query)
    key_codes, key_scale = ternary_activation_codes(key)
    reconstructed_query = query_codes * query_scale
    reconstructed_key = key_codes * key_scale
    expected = (
        reconstructed_query @ reconstructed_key.transpose(-2, -1)
    ) / 8**0.5

    scores, accumulators = ternary_qk_attention_reference(query, key)

    assert accumulators.dtype == torch.int32
    assert accumulators.shape == (2, 3, 4, 5)
    assert accumulators.abs().max().item() <= 8
    assert torch.allclose(scores, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("shared_scales", [False, True])
def test_binary_qk_reference_matches_reconstructed_attention_scores(
    shared_scales: bool,
) -> None:
    torch.manual_seed(29)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 5, 8)
    query_codes, dynamic_query_scale = binary_activation_codes(query)
    key_codes, dynamic_key_scale = binary_activation_codes(key)
    if shared_scales:
        query_scale = torch.rand(1, 3, 1, 1).clamp_min(0.1)
        key_scale = torch.rand(1, 3, 1, 1).clamp_min(0.1)
    else:
        query_scale = None
        key_scale = None
    expected_query_scale = (
        dynamic_query_scale if query_scale is None else query_scale
    )
    expected_key_scale = dynamic_key_scale if key_scale is None else key_scale
    expected = (
        (query_codes * expected_query_scale)
        @ (key_codes * expected_key_scale).transpose(-2, -1)
    ) / 8**0.5

    scores, accumulators = binary_qk_attention_reference(
        query,
        key,
        query_scale=query_scale,
        key_scale=key_scale,
    )

    assert accumulators.dtype == torch.int32
    assert accumulators.shape == (2, 3, 4, 5)
    assert accumulators.abs().max().item() <= 8
    assert torch.allclose(scores, expected, atol=1e-6, rtol=1e-6)


def test_binary_qk_reference_rejects_nonbroadcastable_scales() -> None:
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 5, 8)

    with pytest.raises(ValueError, match="broadcastable"):
        binary_qk_attention_reference(
            query,
            key,
            query_scale=torch.ones(7),
        )


def test_learned_scale_ternarization_updates_shadow_and_scale() -> None:
    values = torch.tensor([[-0.8, -0.1, 0.7]], requires_grad=True)
    log_scale = torch.tensor(0.5).log().requires_grad_()
    quantized, codes = ternarize_activation_with_learned_scale(
        values,
        log_scale.exp(),
    )

    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.equal(quantized.detach(), codes * 0.5)
    quantized.square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert log_scale.grad is not None and log_scale.grad.abs().item() > 0


def test_binary_activation_codes_and_learned_scale_use_exact_signs() -> None:
    values = torch.tensor([[-0.8, -0.1, 0.0, 0.7]], requires_grad=True)
    codes, dynamic_scale = binary_activation_codes(values)
    log_scale = torch.tensor(0.5).log().requires_grad_()
    quantized, learned_codes = binarize_activation_with_learned_scale(
        values,
        log_scale.exp(),
    )

    assert set(codes.unique().tolist()) == {-1.0, 1.0}
    assert dynamic_scale.item() == pytest.approx(values.detach().abs().mean().item())
    assert torch.equal(codes, learned_codes)
    assert torch.equal(quantized.detach(), learned_codes * 0.5)
    quantized.square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert log_scale.grad is not None and log_scale.grad.abs().item() > 0


def test_int2_route_ternary_value_reference_matches_reconstruction() -> None:
    torch.manual_seed(23)
    route_codes = torch.randint(0, 4, (2, 3, 4, 5), dtype=torch.float32)
    route_codes[..., -1] = 3
    values = torch.randn(2, 3, 5, 8)
    value_scale = torch.full((1, 3, 1, 1), 0.5)
    value_codes = ternary_code(values / value_scale)
    probabilities = route_codes / route_codes.sum(dim=-1, keepdim=True)
    expected = probabilities @ (value_codes * value_scale)

    output, accumulators = int2_route_ternary_value_reference(
        route_codes,
        values,
        value_scale,
    )

    assert accumulators.dtype == torch.int32
    assert accumulators.shape == (2, 3, 4, 8, 2)
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)


def test_int2_route_reference_supports_softmax1_empty_route_code() -> None:
    route_codes = torch.tensor([[[[3.0, 1.0]]]])
    values = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value_scale = torch.ones((1, 1, 1, 1))

    output, _ = int2_route_ternary_value_reference(
        route_codes,
        values,
        value_scale,
        empty_route_code=2,
    )

    expected = torch.tensor([[[[0.5, 1.0 / 6.0]]]])
    assert torch.allclose(output, expected)
