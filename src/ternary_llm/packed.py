from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch import Tensor


def pack_ternary_codes(codes: Tensor) -> tuple[Tensor, int]:
    """Pack four {-1, 0, +1} codes per byte and return the original width."""
    if not bool(torch.all((codes == -1) | (codes == 0) | (codes == 1))):
        raise ValueError("codes must contain only -1, 0, or +1")
    width = codes.size(-1)
    padding = (-width) % 4
    if padding:
        codes = F.pad(codes, (0, padding))
    encoded = torch.where(codes == -1, 2, codes).to(torch.uint8)
    grouped = encoded.reshape(*encoded.shape[:-1], -1, 4).to(torch.int16)
    shifts = torch.tensor([0, 2, 4, 6], device=codes.device, dtype=torch.int16)
    packed = torch.sum(grouped << shifts, dim=-1).to(torch.uint8)
    return packed, width


def unpack_ternary_codes(packed: Tensor, width: int) -> Tensor:
    """Unpack the two-bit format into signed int8 ternary codes."""
    if packed.dtype != torch.uint8:
        raise ValueError("packed codes must use uint8 storage")
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.int16)
    values = (packed.to(torch.int16).unsqueeze(-1) >> shifts) & 0b11
    if bool(torch.any(values == 3)):
        raise ValueError("packed tensor contains the reserved code 3")
    codes = torch.where(values == 2, -1, values).to(torch.int8)
    return codes.reshape(*packed.shape[:-1], -1)[..., :width]


def packed_ternary_linear(
    inputs: Tensor,
    packed_weight: Tensor,
    *,
    in_features: int,
    scale: Tensor,
) -> Tensor:
    """Correctness reference for a packed ternary linear operation.

    This portable PyTorch implementation unpacks before reduction. It validates
    the format and numerical result; a fused device kernel is still required for
    a latency win.
    """
    codes = unpack_ternary_codes(packed_weight, in_features)
    weight = codes.to(inputs.dtype) * scale.to(inputs.dtype)
    return F.linear(inputs, weight)


def benchmark(
    *,
    batch: int,
    in_features: int,
    out_features: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    generator = torch.Generator(device=device).manual_seed(17)
    inputs = torch.randn(batch, in_features, device=device, generator=generator)
    codes = torch.randint(
        -1,
        2,
        (out_features, in_features),
        device=device,
        generator=generator,
        dtype=torch.int8,
    )
    scale = torch.rand(out_features, 1, device=device, generator=generator)
    dense_weight = codes.to(inputs.dtype) * scale
    packed, width = pack_ternary_codes(codes)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    for _ in range(5):
        F.linear(inputs, dense_weight)
        packed_ternary_linear(inputs, packed, in_features=width, scale=scale)
    synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        F.linear(inputs, dense_weight)
    synchronize()
    dense_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        packed_ternary_linear(inputs, packed, in_features=width, scale=scale)
    synchronize()
    packed_seconds = time.perf_counter() - start

    dense_bytes = dense_weight.numel() * dense_weight.element_size()
    packed_bytes = packed.numel() * packed.element_size() + scale.numel() * scale.element_size()
    return {
        "device": str(device),
        "batch": batch,
        "in_features": in_features,
        "out_features": out_features,
        "iterations": iterations,
        "dense_weight_bytes": dense_bytes,
        "packed_weight_and_scale_bytes": packed_bytes,
        "compression_ratio": dense_bytes / packed_bytes,
        "dense_seconds": dense_seconds,
        "packed_reference_seconds": packed_seconds,
        "packed_reference_slowdown": packed_seconds / dense_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    result = benchmark(
        batch=args.batch,
        in_features=args.in_features,
        out_features=args.out_features,
        iterations=args.iterations,
        device=torch.device(args.device),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
