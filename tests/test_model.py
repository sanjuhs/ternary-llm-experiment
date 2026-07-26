import pytest
import torch

from ternary_llm.config import (
    VALID_ATTENTION_QUANTIZATIONS,
    VALID_MODES,
    ModelConfig,
)
from ternary_llm.model import TernaryGPT
from ternary_llm.projection import normalized_hadamard
from ternary_llm.train import distillation_loss


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


@pytest.mark.parametrize("encoding", ["residual_binary", "residual_ternary"])
def test_hadamard_residual_planes_compute_loss_gradients_and_stats(
    encoding: str,
) -> None:
    config = ModelConfig(
        vocab_size=300,
        context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_multiplier=2,
        activation_encoding=encoding,  # type: ignore[arg-type]
        activation_planes=2,
        qkv_quantization="ternary",
    )
    model = TernaryGPT(config, "hadamard_progressive").eval()
    inputs = torch.randint(0, config.vocab_size, (2, config.context_length))
    _, loss = model(inputs, inputs)

    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.grad is not None
    stats = model.residual_stats()["aggregate"]
    assert stats["planes"] == 2
    assert stats["normalized_mse"] >= 0


def test_activation_planes_can_be_allocated_by_layer() -> None:
    config = ModelConfig(
        vocab_size=300,
        context_length=8,
        d_model=16,
        n_layers=3,
        n_heads=2,
        ff_multiplier=2,
        activation_encoding="residual_ternary",
        activation_planes=2,
        activation_planes_by_layer=[2, 2, 3],
    )
    model = TernaryGPT(config, "hadamard_progressive")

    assert [block.activation_planes for block in model.blocks] == [2, 2, 3]
    assert [
        block.attention.qkv.activation_planes for block in model.blocks
    ] == [2, 2, 3]


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


def test_gated_rectified_attention_forces_ternary_qkv() -> None:
    config = ModelConfig(
        vocab_size=300,
        context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_multiplier=2,
        qkv_quantization="ternary",
        attention_rectification="qvit",
        attention_gate="binary",
        attention_quantization="score_int2_lut",
    )
    model = TernaryGPT(config, "ternary_weights").eval()
    inputs = torch.randint(0, config.vocab_size, (2, config.context_length))
    _, loss = model(inputs, inputs)

    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    stats = model.attention_stats()["aggregate"]
    for name in ("q", "k", "v"):
        total = sum(
            stats[f"{name}_{label}_fraction"]
            for label in ("negative", "zero", "positive")
        )
        assert total == pytest.approx(1.0)
    assert stats["gate_open_fraction"] == 1.0


def test_learned_head_qkv_scales_receive_gradients_and_report_stats() -> None:
    config = ModelConfig(
        vocab_size=300,
        context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_multiplier=2,
        qkv_quantization="ternary",
        qkv_scale_granularity="learned_head",
        qkv_scale_initial=0.5,
        attention_quantization="score_lut_prob_int2",
    )
    model = TernaryGPT(config, "ternary_weights").eval()
    inputs = torch.randint(0, config.vocab_size, (2, config.context_length))
    _, loss = model(inputs, inputs)

    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    scale_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("qkv_log_scales")
    ]
    assert len(scale_parameters) == config.n_layers
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in scale_parameters
    )
    stats = model.attention_stats()["aggregate"]
    for name in ("q", "k", "v"):
        assert stats[f"{name}_scale_mean"] == pytest.approx(0.5)


def test_attention_and_qk_distillation_compute_student_gradients() -> None:
    from dataclasses import replace

    from ternary_llm.config import DataConfig, ExperimentConfig, TrainConfig

    model_config = ModelConfig(
        vocab_size=300,
        context_length=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_multiplier=2,
        qkv_quantization="ternary",
        attention_rectification="qvit",
        attention_gate="binary",
        attention_quantization="score_int2_lut",
    )
    student = TernaryGPT(model_config, "ternary_weights")
    teacher = TernaryGPT(
        replace(
            model_config,
            qkv_quantization="inherit",
            attention_rectification="none",
            attention_gate="none",
            attention_quantization="float",
        ),
        "ternary_weights",
    )
    student.set_capture_distillation(True)
    teacher.set_capture_distillation(True)
    inputs = torch.randint(0, model_config.vocab_size, (2, model_config.context_length))
    student_logits, _ = student(inputs, inputs)
    with torch.no_grad():
        teacher_logits, _ = teacher(inputs)
    config = ExperimentConfig(
        seed=17,
        mode="ternary_weights",
        device="cpu",
        model=model_config,
        data=DataConfig(train_bin="unused", validation_bin="unused", tokenizer="unused"),
        train=TrainConfig(
            batch_size=2,
            max_steps=1,
            learning_rate=1e-3,
            min_learning_rate=0.0,
            warmup_steps=0,
            weight_decay=0.0,
            grad_clip=1.0,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            output_dir="unused",
            logit_distillation_weight=0.1,
            attention_distillation_weight=0.2,
            qk_distillation_weight=0.3,
            hidden_distillation_weight=0.4,
            distillation_token_stride=2,
        ),
    )
    auxiliary, metrics = distillation_loss(
        student,
        teacher,
        student_logits,
        teacher_logits,
        config,
    )
    auxiliary.backward()

    assert torch.isfinite(auxiliary)
    assert set(metrics) == {
        "logit_distillation_loss",
        "attention_distillation_loss",
        "qk_distillation_loss",
        "hidden_distillation_loss",
    }
    assert student.blocks[0].attention.q_gamma.grad is not None
