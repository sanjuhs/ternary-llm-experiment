import torch

from ternary_llm.export import pack_state_dict, unpack_state_dict
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
