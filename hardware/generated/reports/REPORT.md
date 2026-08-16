# Frozen ternary Transformer hardware feasibility report

This report distinguishes measured checkpoint facts, verified RTL, pre-layout synthesis,
and scenario assumptions. It is **not** a tape-out signoff report.

## Result

- Source artifact SHA-256: `9eae9d27c0a10da53672cb3d711ba634fea14560b96845a789ac660f0402be23`
- Packed deployment size: 7.514 MB
- Ternary scalars: 29,499,904
- Measured weight zeros: 31.62%
- RTL simulation: **passed**
- Strict ternary-operand contract: **True**
- End-to-end integer reference: **False**

The checkpoint can be mapped to a frozen ternary datapath. It is not yet an end-to-end
integer token generator because requantization and the final sampling policy remain
explicit boundaries.

## 45 nm primitive synthesis

Yosys/ABC mapped a 16-lane dot-product primitive into the open Nangate45 typical
standard-cell library. These are combinational, pre-layout results.

| Primitive | Mapped cells | Area (um²) | ABC delay (ps) | Reciprocal delay (MHz) |
|---|---:|---:|---:|---:|
| frozen_ternary | 232 | 234.878 | 419.11 | 2386 |
| programmable_ternary | 485 | 458.318 | 504.89 | 1981 |
| programmable_int8 | 15619 | 15309.896 | 1308.63 | 764 |

The frozen result uses actual codes from
`blocks.0.attention.qkv.weight[0, 192:208]`.
The INT8 result is a programmable signed MAC baseline, so the comparison intentionally
captures both low precision and checkpoint freezing.

- INT8 / frozen-ternary primitive area: 65.2x
- INT8 / programmable-ternary primitive area: 33.4x
- INT8 / frozen-ternary area-normalized throughput: 203.5x
- INT8 / programmable-ternary area-normalized throughput: 86.6x

The first density result crosses two orders of magnitude, but only for this local,
hardwired primitive. It is not a full-chip speedup. The system scenario below instead
uses a reusable programmable ternary core plus SRAM.

## Weight storage

- Versus INT8: 3.95x smaller
- Versus FP16: 7.90x smaller
- Versus FP32: 15.81x smaller

Ternary representation alone therefore does not reach a 100x memory reduction.

## Incremental token workload

- Linear ternary contributions: 81,788,928
- Full-context QK contributions: 1,048,576
- Full-context route-V contributions: 1,048,576
- Hadamard add/subtracts: 82,944
- Total low-bit contributions: 83,969,024
- Packed weight stream: 6.816 MB/token
- Packed full-context K/V cache: 0.524 MB

## Architecture scenario—not a measured chip

At 4,096 contribution lanes,
60% utilization, and
500 MHz:

- 34,168 cycles/token
- 68.34 us/token
- 14,634 raw tokens/s
- 99.7 GB/s internal packed-weight bandwidth
- 15.20 uJ/token under the declared energy coefficients
- 0.22 W estimated active power
- 30.5 mm² scenario SRAM area
- 0.235 mm² scenario contraction area
- 35.3 mm² scenario total die area

These figures assume a real incremental K/V-cache implementation. They must not be
compared to the current PyTorch path, which recomputes the context during generation.

## Illustrative cost scenario

The calculator assumes a 300 mm wafer,
$6,000 wafer cost,
35.3 mm² estimated die, and
0.20 defects/cm². These values are
editable assumptions, not a foundry quote.

- Gross dies/wafer: 1889
- Estimated yield: 87.2%
- Bare-die cost: $3.64
- Packaged/tested unit cost: $7.64

This excludes NRE, masks, SRAM/compiler IP, physical design, verification, boards,
inventory, and engineering. The assumed die area has not been established by place-and-route.

## Evidence ladder

1. **Measured:** artifact checksum, tensor shapes, packed bytes, and code counts.
2. **Verified:** actual frozen-weight RTL passes Icarus Verilog simulation.
3. **Pre-layout:** 16-lane primitives mapped through Yosys/ABC to Nangate45.
4. **Analytical:** model-wide operation, cache, and bandwidth counts.
5. **Scenario only:** full-chip area, clock, energy, tokens/s, yield, and cost.

## Required before claiming silicon performance

- close the fixed-point requantization and final sampling boundaries;
- verify token-for-token equivalence on the complete integer runtime;
- integrate compiled SRAM macros and a banked weight/KV memory system;
- perform floorplan, placement, clock-tree synthesis, routing, extraction, STA, and power analysis;
- prototype on FPGA, then obtain foundry and packaging quotes.
