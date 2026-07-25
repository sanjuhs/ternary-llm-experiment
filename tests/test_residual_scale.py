import torch

from ternary_llm.config import ModelConfig
from ternary_llm.model import TransformerBlock


def test_zero_like_branch_scale_approximately_preserves_residual() -> None:
    torch.manual_seed(17)
    base = ModelConfig(
        vocab_size=32,
        context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        residual_scale=1.0,
    )
    scaled = ModelConfig(**{**base.__dict__, "residual_scale": 1e-6})
    block = TransformerBlock(scaled, "float").eval()
    inputs = torch.randn(2, 8, 16)

    output = block(inputs)
    assert torch.allclose(output, inputs, atol=1e-4)
