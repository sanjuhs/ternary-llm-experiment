import torch

from ternary_llm.quantization import (
    integer_softmax_from_int2_codes,
    population_ternary_codes,
    progressive_activation_codes,
    quantize_activation_a4,
    quantize_activation_levels,
    quantize_attention_probabilities_binary,
    quantize_attention_probabilities_int2,
    quantize_attention_scores_int2,
    ternarize,
    ternarize_activation,
    ternary_code,
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
