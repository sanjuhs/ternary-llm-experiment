import torch

from ternary_llm.projection import (
    CovarianceAccumulator,
    coat_projection,
    covariance_diagonal_cv,
    normalized_hadamard,
)
from ternary_llm.quantization import (
    quantize_activation_a4,
    quantize_projected_activation,
)


def test_normalized_hadamard_is_orthogonal() -> None:
    matrix = normalized_hadamard(16)
    assert torch.allclose(matrix.T @ matrix, torch.eye(16), atol=1e-6)


def test_coat_projection_equalizes_covariance_diagonal() -> None:
    covariance = torch.diag(torch.tensor([1.0, 2.0, 8.0, 32.0]))
    projection = coat_projection(covariance)
    assert covariance_diagonal_cv(covariance, projection) < 1e-6
    assert covariance_diagonal_cv(covariance, projection) < covariance_diagonal_cv(covariance)


def test_covariance_accumulator_matches_torch_cov() -> None:
    generator = torch.Generator().manual_seed(7)
    samples = torch.randn(25, 8, generator=generator)
    accumulator = CovarianceAccumulator(8)
    accumulator.update(samples[:10])
    accumulator.update(samples[10:])
    assert torch.allclose(
        accumulator.covariance(),
        torch.cov(samples.T).to(torch.float64),
        atol=1e-10,
    )


def test_projected_ternary_has_straight_through_gradient() -> None:
    values = torch.randn(2, 4, requires_grad=True)
    projection = normalized_hadamard(4)
    output = quantize_projected_activation(
        values,
        "hadamard_ternary",
        0.5,
        projection,
        lanes=1,
    )
    output.sum().backward()
    assert torch.allclose(values.grad, torch.ones_like(values), atol=1e-6)


def test_a4_has_no_more_than_sixteen_levels_per_vector() -> None:
    values = torch.linspace(-2, 3, 100).reshape(2, 50)
    quantized = quantize_activation_a4(values)
    assert all(row.unique().numel() <= 16 for row in quantized)
