from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parametrize

from ternary_llm.minicpmo_native_qat import attach_dynamic_ternary_qat


def test_dynamic_qat_forward_is_exactly_ternary_scaled() -> None:
    layer = nn.Linear(8, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [-2.0, -0.1, 0.0, 0.1, 2.0, 0.2, -0.2, 1.0],
                    [0.0, 0.0, 3.0, -3.0, 0.1, -0.1, 0.2, -0.2],
                ]
            )
        )
    qat = attach_dynamic_ternary_qat(layer, "weight", group_size=4)
    original = layer.parametrizations.weight.original
    codes = qat.codes(original)
    reconstructed = (codes.float() * qat.scales.detach()).reshape_as(layer.weight)

    assert set(codes.unique().tolist()).issubset({-1, 0, 1})
    torch.testing.assert_close(layer.weight.detach().float(), reconstructed)


def test_dynamic_qat_backpropagates_to_shadow_and_scales() -> None:
    torch.manual_seed(0)
    layer = nn.Linear(4, 3, bias=False)
    qat = attach_dynamic_ternary_qat(layer, "weight", group_size=4)
    output = layer(torch.randn(2, 4)).square().mean()
    output.backward()

    original = layer.parametrizations.weight.original
    assert original.grad is not None
    assert torch.count_nonzero(original.grad) > 0
    assert qat.raw_scales.grad is not None
    assert torch.count_nonzero(qat.raw_scales.grad) > 0


def test_dynamic_codes_can_change_and_export_omits_shadow_weights() -> None:
    layer = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.01, 0.02, 1.0, -1.0]]))
    qat = attach_dynamic_ternary_qat(layer, "weight", group_size=4)
    original = layer.parametrizations.weight.original
    before = qat.codes(original).clone()

    with torch.no_grad():
        original[0, 0] = qat.scales.detach()[0, 0] * 2
    after = qat.codes(original)
    payload = qat.artifact_payload(original)

    assert before[0, 0].item() == 0
    assert after[0, 0].item() == 1
    assert payload["shadow_weights_exported"] is False
    assert "shadow_weights" not in payload


def test_parametrization_can_be_removed_to_materialize_ternary_weight() -> None:
    layer = nn.Linear(4, 2, bias=False)
    attach_dynamic_ternary_qat(layer, "weight", group_size=4)
    expected = layer.weight.detach().clone()
    parametrize.remove_parametrizations(layer, "weight", leave_parametrized=True)

    assert not parametrize.is_parametrized(layer, "weight")
    torch.testing.assert_close(layer.weight.detach(), expected)
