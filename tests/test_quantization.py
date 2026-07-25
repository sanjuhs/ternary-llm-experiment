import torch

from ternary_llm.quantization import (
    population_ternary_codes,
    quantize_activation_a4,
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
