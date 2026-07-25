from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ternary_llm.packed import pack_ternary_codes

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


if triton is not None and tl is not None:

    @triton.jit
    def _packed_ternary_linear_kernel(
        inputs,
        packed_weight,
        scale,
        output,
        rows: tl.constexpr,
        in_features: tl.constexpr,
        out_features: tl.constexpr,
        packed_stride: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        program_m = tl.program_id(0)
        program_n = tl.program_id(1)
        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in range(0, in_features, BLOCK_K):
            offsets_k = start_k + tl.arange(0, BLOCK_K)
            input_block = tl.load(
                inputs + offsets_m[:, None] * in_features + offsets_k[None, :],
                mask=(offsets_m[:, None] < rows) & (offsets_k[None, :] < in_features),
                other=0.0,
            )
            packed_block = tl.load(
                packed_weight
                + offsets_n[:, None] * packed_stride
                + (offsets_k[None, :] // 4),
                mask=(offsets_n[:, None] < out_features)
                & (offsets_k[None, :] < in_features),
                other=0,
            ).to(tl.int32)
            shifts = (offsets_k[None, :] % 4) * 2
            encoded = (packed_block >> shifts) & 3
            codes = tl.where(encoded == 2, -1.0, encoded.to(tl.float32)).to(
                input_block.dtype
            )
            accumulator += tl.dot(input_block, tl.trans(codes))

        output_scale = tl.load(
            scale + offsets_n,
            mask=offsets_n < out_features,
            other=0.0,
        )
        values = accumulator * output_scale[None, :]
        tl.store(
            output + offsets_m[:, None] * out_features + offsets_n[None, :],
            values,
            mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < out_features),
        )


def triton_packed_ternary_linear(
    inputs: Tensor,
    packed_weight: Tensor,
    scale: Tensor,
    *,
    in_features: int,
) -> Tensor:
    if triton is None:
        raise RuntimeError("Triton is unavailable; run this kernel on a CUDA PyTorch install")
    if inputs.device.type != "cuda":
        raise ValueError("the Triton kernel requires CUDA tensors")
    if inputs.ndim != 2 or packed_weight.ndim != 2:
        raise ValueError("inputs and packed_weight must both be matrices")
    rows, actual_in_features = inputs.shape
    if actual_in_features != in_features:
        raise ValueError("input width does not match in_features")
    out_features = packed_weight.size(0)
    packed_stride = packed_weight.size(1)
    output = torch.empty(rows, out_features, device=inputs.device, dtype=inputs.dtype)
    block_m, block_n, block_k = 32, 32, 32
    grid = (triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))
    _packed_ternary_linear_kernel[grid](
        inputs,
        packed_weight,
        scale.reshape(-1),
        output,
        rows,
        in_features,
        out_features,
        packed_stride,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


def benchmark(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    iterations: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(17)
    inputs = torch.randn(
        rows,
        in_features,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    codes = torch.randint(
        -1,
        2,
        (out_features, in_features),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    scale = torch.rand(out_features, 1, device=device, generator=generator)
    packed, width = pack_ternary_codes(codes)
    dense_weight = codes.to(torch.bfloat16) * scale.to(torch.bfloat16)

    expected = F.linear(inputs, dense_weight)
    actual = triton_packed_ternary_linear(
        inputs,
        packed,
        scale,
        in_features=width,
    )
    torch.cuda.synchronize()
    max_error = float((actual - expected).abs().max().item())

    for _ in range(10):
        F.linear(inputs, dense_weight)
        triton_packed_ternary_linear(inputs, packed, scale, in_features=width)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        F.linear(inputs, dense_weight)
    torch.cuda.synchronize()
    dense_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        triton_packed_ternary_linear(inputs, packed, scale, in_features=width)
    torch.cuda.synchronize()
    packed_seconds = time.perf_counter() - start

    dense_bytes = dense_weight.numel() * dense_weight.element_size()
    packed_bytes = packed.numel() + scale.numel() * scale.element_size()
    return {
        "gpu": torch.cuda.get_device_name(device),
        "rows": rows,
        "in_features": in_features,
        "out_features": out_features,
        "iterations": iterations,
        "max_absolute_error_bf16": max_error,
        "dense_weight_bytes": dense_bytes,
        "packed_weight_and_scale_bytes": packed_bytes,
        "compression_ratio": dense_bytes / packed_bytes,
        "dense_milliseconds": dense_seconds * 1000 / iterations,
        "packed_triton_milliseconds": packed_seconds * 1000 / iterations,
        "speedup": dense_seconds / packed_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=16_384)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                rows=args.rows,
                in_features=args.in_features,
                out_features=args.out_features,
                iterations=args.iterations,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
