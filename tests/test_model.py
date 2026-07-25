import pytest
import torch

from ternary_llm.config import VALID_MODES, ModelConfig
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
