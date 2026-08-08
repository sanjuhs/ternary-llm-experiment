from pathlib import Path

import torch

from ternary_llm.hardware import (
    ArchitectureAssumptions,
    architecture_estimate,
    generate_rtl,
    model_workload,
)
from ternary_llm.packed import pack_ternary_codes


def _item(codes: torch.Tensor) -> dict[str, object]:
    packed, width = pack_ternary_codes(codes.to(torch.int8))
    return {
        "kind": "scaled-ternary-2bit-v1",
        "packed_codes": packed,
        "packed_width": width,
        "shape": list(codes.shape),
        "scale": torch.ones((*codes.shape[:-1], 1), dtype=torch.float16),
    }


def _artifact() -> dict[str, object]:
    matrix = torch.tensor([[1, 0, -1, 1], [-1, 1, 0, 0]])
    return {
        "format": "ternary-deployment-v2",
        "config": {
            "model": {
                "context_length": 4,
                "d_model": 4,
                "n_layers": 1,
                "n_heads": 1,
                "activation_planes": 3,
            }
        },
        "model": {
            "token_embedding": _item(torch.ones((8, 4))),
            "blocks.0.attention.qkv.weight": _item(matrix),
            "blocks.0.attention.projection.weight": _item(matrix),
            "blocks.0.feed_forward.up.weight": _item(matrix),
            "blocks.0.feed_forward.down.weight": _item(matrix),
        },
    }


def test_workload_counts_incremental_decode() -> None:
    workload = model_workload(_artifact())  # type: ignore[arg-type]
    assert workload["matrix_weights_per_token"] == 32
    assert workload["vocabulary_weights_per_token"] == 32
    assert workload["linear_ternary_contributions_per_token"] == 192
    assert workload["qk_ternary_contributions_per_token_at_full_context"] == 16
    assert workload["route_v_low_bit_contributions_per_token_at_full_context"] == 16
    assert workload["packed_kv_cache_bytes_at_full_context"] == 8


def test_estimate_is_explicit_scenario() -> None:
    estimate = architecture_estimate(  # type: ignore[arg-type]
        _artifact(),
        ArchitectureAssumptions(contribution_lanes=16, clock_mhz=100.0),
    )
    assert estimate["status"] == "scenario_estimate_not_signoff"
    assert estimate["cycles_per_token"] > 0
    assert estimate["raw_tokens_per_second"] > 0


def test_rtl_contains_actual_checkpoint_codes(tmp_path: Path) -> None:
    artifact = _artifact()
    qkv = torch.tensor([[1, 0, -1, 1], [-1, 1, 0, 0]])
    artifact["model"]["blocks.0.attention.qkv.weight"] = _item(qkv)  # type: ignore[index]
    result = generate_rtl(  # type: ignore[arg-type]
        artifact, tmp_path, lanes=4, representative=False
    )
    source = (tmp_path / "frozen_ternary_dot_4.sv").read_text()
    assert result["selected_weight_codes"] == [1, 0, -1, 1]
    assert "8'b01100001" in source
