import torch

from ternary_llm.config import ModelConfig
from ternary_llm.diagnostics import activation_diagnostics
from ternary_llm.model import TernaryGPT


def test_activation_diagnostics_capture_transformer_boundaries() -> None:
    config = ModelConfig(
        vocab_size=32,
        context_length=8,
        d_model=16,
        n_layers=2,
        n_heads=2,
    )
    model = TernaryGPT(config, "ternary_forward").eval()
    result = activation_diagnostics(model, torch.randint(0, 32, (2, 8)))

    assert "blocks.0" in result["boundaries"]
    assert "blocks.1" in result["boundaries"]
    assert len(result["residual_sign_change"]) == 1
    assert result["mean_output_entropy"] > 0
