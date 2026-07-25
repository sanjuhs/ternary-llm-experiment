# Complete Transformer quantization contract

“The whole Transformer is quantized” is ambiguous unless we say what crosses
each storage and compute boundary. This project uses the following strict but
implementable definition.

## What must be low-bit

| Boundary | Strict target | Current experiment |
|---|---:|---:|
| Embedding and linear weights | ternary, packed in 2 bits | ternary fake quantization; packed export exists |
| Residual-stream activations | ternary or INT4 | COAT/Hadamard ternary and A4 variants |
| Q, K, and V operands | ternary | independently forced by `qkv_quantization = "ternary"` |
| Softmax input | four codes (2 bits) | `score_int2` |
| Normalized attention routes | four codes or one bit | `prob_int2`, `prob_binary` |
| KV cache | same low-bit format as K/V | planned runtime experiment |
| Model output between blocks | ternary or INT4 | quantized residual boundary |

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
