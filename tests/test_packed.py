import torch
import torch.nn.functional as F

from ternary_llm.packed import (
    pack_ternary_codes,
    packed_ternary_linear,
    unpack_ternary_codes,
)


def test_pack_round_trip_for_non_multiple_of_four() -> None:
    codes = torch.tensor([[-1, 0, 1, -1, 1, 0, 0]], dtype=torch.int8)
    packed, width = pack_ternary_codes(codes)

    assert packed.numel() == 2
    assert torch.equal(unpack_ternary_codes(packed, width), codes)


def test_packed_linear_matches_dense_ternary_linear() -> None:
    inputs = torch.randn(3, 7)
    codes = torch.randint(-1, 2, (5, 7), dtype=torch.int8)
    scale = torch.rand(5, 1)
    packed, width = pack_ternary_codes(codes)

    actual = packed_ternary_linear(inputs, packed, in_features=width, scale=scale)
    expected = F.linear(inputs, codes.to(inputs.dtype) * scale)
    assert torch.allclose(actual, expected)
