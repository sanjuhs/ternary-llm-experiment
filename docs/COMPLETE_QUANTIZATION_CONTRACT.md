# Complete Transformer quantization contract

“The whole Transformer is quantized” is ambiguous unless we say what crosses
each storage and compute boundary. This project uses the following strict but
implementable definition.

## What must be low-bit

| Boundary | Strict target | Current experiment |
|---|---:|---:|
| Embedding, normalization, and linear weights | ternary, packed in 2 bits | ternary fake quantization plus deployment export |
| RMSNorm arithmetic | fixed-point input, integer reductions and square root | opt-in `integer_reference` runtime with STE training |
| Residual-stream activations | binary/ternary code planes | fixed-Hadamard residual-plane variants |
| Q, K, and V operands | ternary, permitting binary as a strict subset | `ternary`, plus matched `binary_qk_ternary_v` fallback |
| Softmax input | four codes (2 bits) | `score_lut_prob_int2` |
| Normalized attention routes | four codes (2 bits) | integer-LUT route quantization |
| KV cache | ternary K/V codes plus shared head scales | learned-head-scale comparison |
| Feed-forward nonlinearity | comparison/zeroing only | matched GELU-versus-ReLU hardening |
| Model output between blocks | binary/ternary code planes | quantized residual boundary |

The repository includes exact arithmetic references for both linear projections
and attention Q·K. Separate Q·K references convert Q and K into exact ternary
or binary code tensors, perform the large dot product with INT32 accumulation,
and apply the two small scale factors afterward. The binary reference covers
both magnitude-optimal per-vector scales and factorized learned per-head
scales. Tests compare these paths against the reconstructed fake-quantized
tensors element for element.

Route·V is not yet equally clean in the current quality path. V is ternary-coded
but has a separate scale per token; because attention mixes many token
positions, those scales cannot be pulled outside the whole reduction as one
factor. The ASIC-strict follow-up will compare a learned layer/head-shared V
scale, following BWTA, against fixed-point scale multipliers and scale-bucketed
value planes. Until one of those paths is validated, “ternary V operand” is
true, while “pure binary-by-ternary Route·V kernel” remains a target.

The learned head-shared branch is now executable. Each attention block stores
three small vectors of positive scales—one Q, K, and V value per head. The
large tensors remain ternary codes; Q·K and Route·V use the packed codes, and
the corresponding scale products can be applied after accumulation. This is an
opt-in experiment and does not alter the running per-token-scale baseline. It
is evaluated against an equal-budget per-token control.

An exact Route·V arithmetic reference accompanies it. Each `{0,1,2,3}` route
code is split into low and high binary planes. Both planes multiply ternary V
codes into INT32 accumulators; the high-plane result is shifted once, the two
planes are added, the integer route-count denominator is applied, and the one
shared V scale reconstructs the output. Tests compare this result with the
explicit reconstructed tensors.

## What the deployment export proves

`ternary-export` now emits `ternary-deployment-v2`. It does not blindly call
every small parameter “ternary”:

- matrix, embedding, normalization, and fixed-Hadamard tensors use packed
  two-bit ternary codes plus their row/tensor scale;
- learned positive Q/K/V head scales use explicit INT16 fixed-point codes;
- Boolean readiness buffers are preserved losslessly;
- the artifact records an inference-contract checklist and any violations.

Every completed downstream arm produces this export and its metadata. This
prevents a per-token-scale or GELU checkpoint from being mislabeled as the
final ASIC-ready endpoint merely because its shadow weights can be packed.

## What cannot literally stay in two bits

Dot products sum hundreds of products. Softmax also sums a row of exponentials.
Those accumulators need more than two bits or they overflow almost immediately.
Accordingly, the strict contract permits:

- INT32 or wider accumulators inside matrix multiplication;
- a wider integer row sum inside softmax;
- floating-point master weights, gradients, optimizer state, and loss during
  quantization-aware training;
- scale factors and normalization statistics stored at higher precision.

This is not an exception invented for this project. Integer accelerators
normally multiply low-bit operands into wide accumulators and requantize at the
next boundary. The meaningful systems claim is that large stored tensors and
matrix-multiply operands are low-bit—not that every temporary scalar is.

An INT32 accumulator does not force FP32 output. The accumulator is immediately
combined with fixed-point or power-of-two scales, rounded, clamped, and stored at
the next ternary or INT4 boundary. In this model, Q·K needs about seven signed
bits, width-256 linear reductions about ten, and width-1,024 feed-forward
reductions about twelve. INT32 is a convenient implementation container, not the
minimum ASIC width.

The present end-to-end PyTorch path is still not a fused integer runtime. Its
portable RMSNorm reference quantizes the input to fixed-point codes, accumulates
squares in INT64, uses a tensorized exact integer square root and division, and
applies ternary normalization weights. The `integer_reference` model option now
wires that arithmetic through both Transformer norms and the final norm.
Training uses its exact forward result with the ordinary RMSNorm derivative as
a straight-through surrogate. Tests cover exact arithmetic, model gradients,
checkpoint serialization, and checkpoint-level evaluation. A same-batch smoke
comparison changed loss from 6.19863 to 6.19745; this is a wiring check, not a
TinyStories quality result. Scale/requantization arithmetic and the final
token-sampling softmax remain explicit deployment boundaries. The option must
still pass matched exhaustive validation on the 27.4M endpoint before the
project can call that boundary quality-preserving.

## Matched attention experiment

All rows use the same checkpoint, validation samples, and random seed.

| Arm | Q/K/V and residuals | Attention representation | Question |
|---|---|---|---|
| A | COAT A4 | float | positive control |
| B | COAT A4 | 2-bit shifted score | Can a four-entry exponential input preserve loss? |
| C | COAT A4 | 2-bit probability | Can routing itself use only four levels? |
| D | COAT A4 | 2-bit score and 2-bit probability | Can the complete attention path be low-bit? |
| E | COAT A4 | binary probability | How much does one-bit routing cost? |
| F | COAT ternary | best of B–E | Does the attention method rescue full ternary activations? |

Arm B quantizes each causal attention row after subtracting its maximum, using
codes `{-3, -2, -1, 0}`. Arm C quantizes each normalized row to codes
`{0, 1, 2, 3}` and renormalizes. Arm D applies both quantizers. Arm E keeps
entries above a fraction of the row maximum, then renormalizes the surviving
routes.

## Success criteria

The primary number is validation cross-entropy on the full TinyStories
validation stream. We also record perplexity, attention entropy, code
utilization, zero fraction, generated samples, gradient norm, and throughput.
A variant is “quality-preserving” only if the result repeats across at least
three seeds and the confidence interval is practically close to its matched
control. One short pilot can reject a bad method; it cannot establish parity.
