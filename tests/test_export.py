import torch

from ternary_llm.export import (
    inference_contract,
    pack_deployment_state_dict,
    pack_state_dict,
    unpack_deployment_state_dict,
    unpack_state_dict,
)
from ternary_llm.quantization import ternarize_weight


def test_packed_state_round_trip_matches_fake_quantized_weights() -> None:
    weights = {
        "matrix": torch.tensor([[-0.8, -0.1, 0.7], [0.2, 0.5, -0.9]]),
        "vector": torch.tensor([-0.5, 0.0, 0.8]),
    }
    packed = pack_state_dict(weights, threshold=0.5)
    restored = unpack_state_dict(packed)

    for name, tensor in weights.items():
        expected = ternarize_weight(tensor, threshold=0.5)
        assert torch.allclose(restored[name], expected, atol=5e-4)


def test_deployment_pack_preserves_fixed_point_scales_and_raw_buffers() -> None:
    state = {
        "blocks.0.attention.qkv.weight": torch.tensor(
            [[-0.8, -0.1, 0.7], [0.2, 0.5, -0.9]]
        ),
        "blocks.0.attention.qkv_log_scales": torch.tensor(
            [[-1.2, -0.7], [-0.5, -0.2], [-1.0, 0.1]]
        ),
        "blocks.0.coat_projection_ready": torch.tensor(True),
    }

    packed = pack_deployment_state_dict(state, threshold=0.5)
    restored = unpack_deployment_state_dict(packed)

    assert packed["blocks.0.attention.qkv.weight"]["kind"] == (
        "scaled-ternary-2bit-v1"
    )
    assert packed["blocks.0.attention.qkv_log_scales"]["kind"] == (
        "positive-scale-int16-v1"
    )
    assert packed["blocks.0.coat_projection_ready"]["kind"] == "raw-buffer-v1"
    assert torch.allclose(
        restored["blocks.0.attention.qkv.weight"],
        ternarize_weight(state["blocks.0.attention.qkv.weight"]),
        atol=5e-4,
    )
    assert torch.allclose(
        restored["blocks.0.attention.qkv_log_scales"],
        state["blocks.0.attention.qkv_log_scales"],
        atol=1e-4,
    )
    assert torch.equal(
        restored["blocks.0.coat_projection_ready"],
        state["blocks.0.coat_projection_ready"],
    )


def test_inference_contract_distinguishes_operands_from_end_to_end_runtime() -> None:
    config = {
        "mode": "hadamard_progressive",
        "model": {
            "activation_encoding": "residual_ternary",
            "qkv_quantization": "ternary",
            "qkv_scale_granularity": "learned_head",
            "attention_quantization": "score_lut_prob_int2",
            "attention_rectification": "none",
            "attention_gate": "none",
            "feed_forward_activation": "relu",
            "dropout": 0.0,
        },
    }

    contract = inference_contract(config)

    assert contract["ternary_operand_contract"]["satisfied"]
    assert not contract["end_to_end_integer_reference"]["satisfied"]
    assert "RMSNorm reciprocal-square-root and scale application" in (
        contract["end_to_end_integer_reference"]["remaining_boundaries"]
    )


def test_inference_contract_rejects_dynamic_scales_and_gelu() -> None:
    config = {
        "mode": "hadamard_progressive",
        "model": {
            "activation_encoding": "residual_ternary",
            "qkv_quantization": "ternary",
            "qkv_scale_granularity": "token",
            "attention_quantization": "score_lut_prob_int2",
            "attention_rectification": "none",
            "attention_gate": "none",
            "feed_forward_activation": "gelu",
            "dropout": 0.0,
        },
    }

    operand_contract = inference_contract(config)["ternary_operand_contract"]

    assert not operand_contract["satisfied"]
    assert set(operand_contract["violations"]) == {
        "factorizable_shared_qkv_scales",
        "integer_friendly_ffn_activation",
    }
