# Ternary LLM experiment ledger and roadmap

*Updated July 25, 2026*

This is the compact, auditable answer to three questions:

1. What have we already done?
2. What is the best result so far?
3. What are we doing next to make attention and eventually the whole inference
   graph low-bit?

## The headline

The best full-corpus model remains the floating-point baseline at **1.6989
validation loss**. The best full-corpus model with ternary weights is **2.0312**.
On the shorter deterministic evaluation used for the projection study, the same
ternary-weight checkpoint measured **2.0251**. Those two numbers use different
validation scopes and must not be presented as a direct improvement.

The best result with ternary weights and 4-bit residual activations is **2.1126**.
The best result with a 2-bit attention representation is **2.1383**. A model with
one ternary code at every residual boundary has not yet worked: the best such
result is **6.3798**.

## What an INT32 accumulator really means

It does **not** mean the next layer must be FP32.

If both operands are ternary, every product is `-1`, `0`, or `+1`. A dot product
must still sum many products, so its temporary sum needs more bits. For a
length-\(K\) dot product, a signed accumulator needs roughly
\(\lceil\log_2(K+1)\rceil+1\) bits:

| Reduction | Length in this model | Approximate signed bits required |
|---|---:|---:|
| One attention-head Q·K | 32 | 7 |
| Width-256 linear input | 256 | 10 |
| Width-1,024 feed-forward input | 1,024 | 12 |

INT32 is a convenient, safe implementation type. It is not a claim that the
network has become 32-bit. An ASIC could choose narrower accumulators after
range analysis.

The inference dataflow is:

```text
ternary codes × ternary codes
        ↓
wider integer accumulator
        ↓
fixed-point scale / shift / bias
        ↓
round and clamp
        ↓
next ternary or INT4 boundary
```

The next stored tensor can therefore be ternary, INT4, or another chosen low-bit
format. No FP32 tensor is required between matrix multiplications.

Softmax, normalization, and residual addition have the same principle but
different reductions:

| Operation | Low-bit operands | Wider temporary | Output boundary |
|---|---|---|---|
| Q·K | ternary Q and K | INT accumulator | 2-bit score code |
| Integer softmax | 2-bit score code | integer exponential LUT and row sum | 2-bit or 1-bit route |
| Route·V | low-bit route and ternary V | INT accumulator | ternary/INT4 context |
| RMSNorm | low-bit hidden state | sum-of-squares and reciprocal-root state | ternary/INT4 normalized state |
| Residual add | low-bit branch and residual | guard-bit integer add | ternary/INT4 residual |

Scale factors, denominators, and accumulator registers occupy a tiny fraction of
the storage and bandwidth used by weights, activations, and the KV cache. For a
future ASIC, learned scales should be constrained to fixed-point or powers of two
so scaling becomes an integer multiply or shift.

## Completed experiments

All result tables below are from this repository's recorded metrics.

### Data and model foundation

- Downloaded the complete `roneneldan/TinyStories` dataset.
- Prepared 2,119,719 training stories, 488,174,163 training tokens, and
  4,907,807 validation tokens.
- Trained a 4,096-token byte-level BPE tokenizer.
- Built a 5,836,032-parameter, six-layer decoder with width 256, eight heads,
  context 256, and reproducible configs.
- Ran one nominal full-corpus pass for the float and ternary-weight models.

### Full-corpus baselines

| Representation | Tokens sampled | Validation loss | Perplexity |
|---|---:|---:|---:|
| Float | 488,177,664 | **1.6989** | 5.47 |
| Ternary weights, float activations | 488,177,664 | **2.0312** | 7.62 |

The ternary-weight model uses about 35.4% zero weight codes. Its packed
inference export is 1,521,077 bytes. This is the strongest evidence so far that
ternary storage and add/subtract/skip linear operators can produce useful
language.

### Strict activation attempts

| Experiment | Validation loss | Perplexity | Outcome |
|---|---:|---:|---|
| Ternary activations, 1,000-step gate | 6.053 | 425.54 | rejected |
| Ternary weights + activations, 1,000-step gate | 6.047 | 422.87 | rejected |
| Two population lanes | 6.2206 | 503.01 | rejected |
| Four population lanes | 6.1606 | 473.72 | rejected |
| Strict ternary with residual scale 0.25 | 6.0255 | 413.83 | rejected |
| Four lanes with residual scale 0.25 | 7.3083 | 1,492.59 | rejected |

The failure signature was a residual stream that became large and stopped
changing sign. This is an architectural problem, not merely a rounding problem.

### Alternative ternary training and storage

- Implemented signed INT8 evidence-counter training. It learned on the smoke
  model but remained oscillatory.
- Implemented stochastic ternary code transitions with only per-row scale state.
  It showed a learning signal but was not ready for a full run.
- Packed four two-bit ternary codes into each byte and verified exact reference
  output.
- Built a fused Triton kernel that consumes packed weights without materializing
  them. It achieved the storage target but was slower than cuBLAS BF16, so the
  kernel performance milestone remains open.

### COAT projection experiments

The COAT-inspired projection reduced residual variance coefficient of variation
from 0.093699 to \(7.0\times10^{-9}\), with maximum orthogonality error
\(7.2\times10^{-7}\).

| Activation path | Evaluation | Validation loss | Perplexity |
|---|---|---:|---:|
| Ternary weights, A16 | 10-batch PTQ | **2.0251** | 7.58 |
| Hadamard A4 | 10-batch PTQ | 2.1712 | 8.77 |
| COAT A4 | 10-batch PTQ | 2.1690 | 8.75 |
| Hadamard ternary | 10-batch PTQ | 8.6642 | 5,791.53 |
| COAT ternary | 10-batch PTQ | 8.6724 | 5,839.59 |
| Hadamard A4 | 1,000-step QAT, 100 batches | 2.1156 | 8.29 |
| COAT A4 | 1,000-step QAT, 100 batches | **2.1126** | 8.27 |
| Hadamard ternary | 1,000-step QAT, 100 batches | 6.3967 | 599.88 |
| COAT ternary | 1,000-step QAT, 100 batches | **6.3798** | 589.82 |

COAT works as a rotation and slightly helps A4. It cannot recover information
after every scalar is collapsed to a single ternary code.

### Two-bit attention experiments

| Attention representation after 500-step QAT | Loss | Perplexity |
|---|---:|---:|
| Float attention control | 2.113237 | 8.2750 |
| 2-bit shifted score, clip 8 | 2.148911 | 8.5755 |
| 2-bit probability | **2.138310** | 8.4851 |
| 2-bit score + 2-bit probability | 2.156921 | 8.6445 |
| Binary route, threshold 0.0625 | 2.208475 | 9.1018 |
| Binary route, threshold 0.125 | 2.188927 | 8.9256 |

This is promising: attention routing can use two-bit codes with a loss increase
of about 0.025 over the matched control. It did not yet force Q, K, and V to
ternary.

### New ternary-QKV pre-training screen

The new code can force Q, K, and V to ternary independently of the residual
format. On ten deterministic batches, applying this post-training to the best
2-bit-probability checkpoint gave:

| New treatment | Loss | Perplexity |
|---|---:|---:|
| Ternary Q/K/V, no rectification | 3.5965 | 36.47 |
| Q-ViT rectification + ternary Q/K/V | 4.1656 | 64.43 |
| Rectified ternary Q/K/V + integer score LUT | 4.1938 | 66.28 |
| Same + 2-bit normalized route | 3.9111 | 49.95 |
| Same + binary route | 4.5143 | 91.31 |

This is a deliberately harsh PTQ screen. It says ternary Q/K/V cannot simply be
switched on after training. It does **not** reject QAT, distillation, or training
the architecture from scratch. Q-ViT's rectification parameters are untrained in
this screen.

## What the four requested papers contribute

### BWTA

[BWTA](https://arxiv.org/abs/2604.03957) contributes binary weights, ternary
activations, a gradual level-reduction schedule, magnitude alignment between
stages, multiple distillation targets, and binary/ternary GPU kernels for linear
and attention operators. Its attention path is the closest published template
for ternary Q/K/V and binary routing.

The limitation is important: its LLM evaluation quantizes only the 30% least
sensitive layers and keeps nonlinear operations and boundary layers at higher
precision. It does not prove that every LLM block can be one-code ternary.

### EXAQ

[EXAQ](https://openreview.net/forum?id=AuJ6gDjcZK) shows that the shifted softmax
input can be clipped and represented with two bits. Four codes permit a
four-entry exponential table. It attacks the exponential and denominator
bottleneck but does not prove two-bit normalized probabilities.

### I-LLM

[I-LLM](https://arxiv.org/abs/2405.17849) provides the strongest evidence that
inference can remain integer-only across linear and nonlinear operations. It
uses dynamic integer matrix multiplication, clips shifted softmax inputs to a
finite window, implements exponentials with shifts, and supplies integer
normalization. Its successful target is W4A4, with 8-bit nonlinear activations.
It is an implementation blueprint, not evidence for W1.58A1.58.

### Q-ViT

[Q-ViT](https://arxiv.org/abs/2210.06707) identifies Q/K/V and attention-map
quantization as the most damaging 2-bit region. Its Information Rectification
Module learns affine transformations that make Q and K quantize more
informatively. Distribution Guided Distillation matches the teacher's Q-Q and
K-K similarity structures rather than only copying the final attention matrix.

Although this is a vision paper, those two training ideas transfer cleanly to a
TinyStories decoder and are now implemented as an ablation.

## The new combined architecture

The pilot does not pretend the papers are interchangeable. It assigns each one a
specific job:

```text
COAT A4 residual input
        │
        ├── ternary-weight QKV projection
        │
        ├── Q-ViT affine rectification of Q and K
        │
        ├── forced ternary Q, K, and V
        │
        ├── ternary Q·K with wider integer accumulation
        │
        ├── EXAQ/I-LLM-inspired 2-bit score code
        │
        ├── four-entry Q15 exponential LUT + integer row sum
        │
        ├── 2-bit or BWTA-style binary route
        │
        ├── route·ternary-V accumulation
        │
        ├── quantizable per-head no-op gate
        │
        └── requantize to the COAT A4 residual boundary
```

The gate uses ternary input and ternary gate weights. At inference its output is
binary, so applying it is a keep/zero operation. During QAT a sigmoid surrogate
passes gradients. It gives an attention head an explicit way to do nothing,
instead of manufacturing extreme softmax logits.

The current implementation is a numerically exact fake-quantization reference.
PyTorch still executes some emulated integer operations in floating tensors. An
honest “integer-only runtime” result requires packed kernels or RTL that consumes
the codes directly.

## The next experiments

### Gate 1: Recover ternary Q/K/V

Run equal-budget 750-step QAT arms:

1. COAT A4 + 2-bit probability control.
2. Forced ternary Q/K/V.
3. Ternary Q/K/V + Q-ViT rectification + binary no-op gate.
4. The same model with 2-bit score LUT and 2-bit normalized routes.

The first target is loss below 2.50. If plain ternary Q/K/V remains above 3.0,
we should not spend a full corpus pass on it.

### Gate 2: Add structured distillation

Use the existing COAT A4 model as teacher. Optimize:

- next-token cross-entropy;
- temperature-scaled logit KL;
- attention-probability MSE, from BWTA;
- sampled Q-Q and K-K similarity MSE, from Q-ViT.

Run three seeds for any arm within 0.10 loss of its matched control.

### Gate 3: Smooth multistage reduction

If direct ternary Q/K/V QAT fails, train through activation alphabets
`19 → 15 → 11 → 7 → 3`, with magnitude alignment at every transition. Keep each
stage only when validation loss recovers before moving to the next.

### Gate 4: Quantize the remaining residual path

Move one boundary at a time:

1. A4 residual + ternary Q/K/V;
2. A4 residual + fully low-bit attention;
3. learned power-of-two residual scales;
4. INT3, then two-plane ternary residual;
5. single-plane ternary residual only if the earlier gates pass.

The residual-free architecture is a separate from-scratch arm, not a drop-in
checkpoint conversion. The 2026 [residual-free
study](https://arxiv.org/abs/2605.25880) shows better W8A8/W8A6 robustness, but
does not provide ternary evidence.

### Gate 5: Prove the ASIC contract

For every surviving arm:

- export packed ternary Q/K/V and KV cache;
- count every FP, integer multiply, shift, add, lookup, and memory byte;
- replace the Python Q15 lookup reference with a fused kernel;
- verify bit-exact outputs against the reference;
- synthesize accumulator widths from observed and worst-case ranges;
- report end-to-end latency and energy, not only kernel speed.

## Decision rule

“Same loss” will mean a predeclared tolerance against a matched control, on the
same validation tokens, across at least three seeds. The first practical target
is within **0.05 cross-entropy** of the COAT A4 control. The stretch target is
within **0.05** of the ternary-weight A16 model. Until then, the architecture is
promising research—not a solved fully ternary Transformer.
