import pytest
import torch

from ternary_llm.config import (
    VALID_ATTENTION_QUANTIZATIONS,
    VALID_MODES,
    ModelConfig,
)
from ternary_llm.model import TernaryGPT
from ternary_llm.projection import normalized_hadamard


@pytest.mark.parametrize("mode", VALID_MODES)
def test_all_forward_modes_compute_loss_and_gradients(mode: str) -> None:
    config = ModelConfig(
        vocab_size=300,
        context_length=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ff_multiplier=2,
    )
    model = TernaryGPT(config, mode)  # type: ignore[arg-type]
    if mode.startswith("coat_"):
        model.set_activation_projection(normalized_hadamard(config.d_model))
    inputs = torch.randint(0, config.vocab_size, (2, config.context_length))
    logits, loss = model(inputs, inputs)

    assert logits.shape == (2, config.context_length, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.grad is not None


def test_generation_respects_requested_length() -> None:
    config = ModelConfig(
        vocab_size=300,
        context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_multiplier=2,
    )
    model = TernaryGPT(config)
    prompt = torch.tensor([[1, 2, 3]])
    generated = model.generate(prompt, max_new_tokens=4, top_k=10)
    assert generated.shape == (1, 7)


@pytest.mark.parametrize("lanes", [2, 4])
def test_population_mode_computes_loss_and_gradients(lanes: int) -> None:
    config = ModelConfig(
        vocab_size=300,
        context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_multiplier=2,
        population_lanes=lanes,
    )
    model = TernaryGPT(config, "population_ternary")
    inputs = torch.randint(0, config.vocab_size, (2, config.context_length))
    _, loss = model(inputs, inputs)

    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.grad is not None


@pytest.mark.parametrize("scheme", VALID_ATTENTION_QUANTIZATIONS)
def test_attention_quantization_schemes_compute_gradients(scheme: str) -> None:
    config = ModelConfig(
        vocab_size=300,
        context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_multiplier=2,
        attention_quantization=scheme,  # type: ignore[arg-type]
    )
    model = TernaryGPT(config, "coat_a4").eval()
    model.set_activation_projection(normalized_hadamard(config.d_model))
    inputs = torch.randint(0, config.vocab_size, (2, config.context_length))
    _, loss = model(inputs, inputs)

    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.grad is not None
    assert model.attention_stats()["aggregate"]


def test_causal_attention_does_not_read_future_tokens() -> None:
    config = ModelConfig(
        vocab_size=300,
        context_length=4,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_multiplier=2,
        attention_quantization="score_int2",
    )
    model = TernaryGPT(config).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 9, 8, 7]])

    logits_a, _ = model(first)
    logits_b, _ = model(changed_future)
    assert torch.allclose(logits_a[:, 0], logits_b[:, 0])
