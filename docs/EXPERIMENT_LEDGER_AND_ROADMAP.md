# Ternary LLM experiment ledger and roadmap

*Updated July 26, 2026*

This is the compact, auditable answer to three questions:

1. What have we already done?
2. What is the best result so far?
3. What are we doing next to make attention and eventually the whole inference
   graph low-bit?

## The headline

The new 27.4M-parameter float control reaches **1.344198 validation loss** and
**3.8351 perplexity** over all 4,907,776 validation targets. Its matched
ternary-weight model reaches **1.535463** and **4.6435**. This replaces the old
5.84M headline values of 1.6989 float and 2.0312 ternary-weight loss.

On common raw validation text, the new float model scores approximately
**0.495229 bits per UTF-8 byte**, 10.5% better than the released
TinyStories-33M checkpoint. The ternary-weight model scores approximately
**0.565695 BPB**, only 2.2% worse than that released reference, although it
remains 14.2% worse than its own matched float teacher.

The best result with ternary weights and 4-bit residual activations is **2.1126**.
The best result with a 2-bit attention representation is **2.1383**. A model with
forced ternary Q/K/V is **2.289840**; adding the strict two-bit score and
four-entry integer-LUT route gives **2.299876**. A model with one ternary code at
every residual boundary now runs and generates, but is not competitive: the best
strict result is **5.098914** (perplexity **163.8439**), versus 6.3798 before the
progressive curriculum.

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

| Model | Representation | Tokens sampled | Exhaustive loss | Perplexity |
|---|---|---:|---:|---:|
| Earlier 5.84M | Float | 488,177,664 | **1.68961** | 5.4174 |
| Earlier 5.84M | Ternary weights, float activations | 488,177,664 | **2.0312** | 7.62 |
| Matched 27.4M | Float | 976,355,328 | **1.344198** | 3.8351 |
| Matched 27.4M | Ternary weights, float activations | 488,177,664 | **1.535463** | 4.6435 |

The new ternary-weight model uses 35.05% zero weight codes. Its packed
inference export is 7,509,079 bytes, versus 337,312,859 bytes for the resumable
shadow-weight checkpoint: a 44.9x file-size reduction. The packed file stores
ternary codes and scales, not optimizer or shadow-weight state.

### Matched 27.4M activation control

The calibrated COAT A4 stage starts from the matched ternary-weight checkpoint
and receives 15,000 adaptation steps with the float model as teacher.

| Representation | Exhaustive loss | Perplexity | Delta from ternary weights |
|---|---:|---:|---:|
| Ternary weights, float activations | 1.535463 | 4.6435 | — |
| Ternary weights, COAT A4 activations | **1.800118** | **6.0504** | +0.264656 |

This is a negative quality result: calibrated four-bit activation boundaries
do not preserve the matched ternary-weight loss at this capacity and training
budget. The checkpoint remains the required control and initialization for the
strict ternary-QKV, integer-attention, multi-plane residual stage.

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

### Progressive fully ternary residual experiment

We then held the already validated low-bit attention path fixed and reduced
every residual boundary through odd symmetric alphabets. Each stage started from
the preceding checkpoint and received magnitude alignment plus quantization-aware
training. The final variants changed GELU to ReLU and attention routing to binary
so the strict inference boundary contains no general activation lookup or
high-precision attention-probability tensor.

| Residual alphabet after staged QAT | Activation / route | Loss | Perplexity |
|---:|---|---:|---:|
| 19 levels | GELU / low-bit | 2.397499 | 10.9956 |
| 15 levels | GELU / low-bit | 2.510083 | 12.3060 |
| 11 levels | GELU / low-bit | 2.764958 | 15.8784 |
| 9 levels | GELU / low-bit | 2.993107 | 19.9476 |
| 7 levels | GELU / low-bit | 3.362680 | 28.8664 |
| 5 levels | GELU / low-bit | 4.236597 | 69.1720 |
| 3 levels | GELU / low-bit | 5.194696 | 180.3133 |
| 3 levels | ReLU / low-bit | 5.148384 | 172.1530 |
| 3 levels | ReLU / binary | 5.142406 | 171.1270 |
| 3 levels, +1,500 CE-only steps | ReLU / binary | **5.098914** | **163.8439** |

The curriculum improves the previous one-code COAT result by **1.2809 loss**,
and the final residual diagnostic confirms exactly three used levels. Residual
codes are 26.46% zero and 73.54% nonzero. The binary attention route is 62.17%
zero and 37.83% one. Gradients stayed finite.

The main scientific finding is a sharp capacity transition below seven levels.
Additional training at three levels helps slowly, but does not close the gap.
Generated text contains TinyStories-like fragments while losing grammar and
long-range coherence. This is a successful execution test and a negative quality
result, not a solved fully ternary language model.

The implementation remains a PyTorch fake-quantization reference. Large stored
operands and persistent boundaries use the declared codes, while dot products,
row sums, normalization statistics, and scaling use wider temporary state. The
next layer does not become FP32 merely because a wider accumulator was used.

### Residual-plane matched screen

The one-code bottleneck was then replaced with successive low-bit residual
refinement planes. Every arm retained ternary weights, ternary Q/K/V, the
two-bit attention-score path, and the four-entry integer exponential lookup.
Each received the same 1,500-step QAT budget.

| Projection | Residual representation | Physical code bits / scalar | Final loss | Perplexity | Residual normalized MSE |
|---|---|---:|---:|---:|---:|
| Learned COAT | 2 binary planes | **2** | 4.339853 | 76.6963 | 0.126216 |
| Fixed Hadamard | 2 binary planes | **2** | **4.332306** | **76.1196** | 0.124672 |
| Learned COAT | 2 ternary planes | 4 | 3.295858 | 27.0006 | 0.043037 |
| Fixed Hadamard | 2 ternary planes | 4 | **3.288210** | **26.7948** | 0.043436 |
| Learned COAT | 3 ternary planes | 6 | 2.664104 | 14.3551 | 0.016615 |
| Fixed Hadamard | 3 ternary planes | 6 | **2.661332** | **14.3153** | 0.016783 |

“Logical bits” such as \(2\log_2(3)=3.17\) are an entropy measure. Ordinary
hardware packs each ternary plane into two bits, so the honest uncompressed
physical costs are four and six bits. The two-binary-plane arm is the only
exact-two-bit activation representation in this table.

The fixed Hadamard transform won every matched trained comparison, despite the
three-plane learned-COAT arm having the better post-training starting point.
This rejects the idea that a dense learned rotation is the key remaining
bottleneck. Quality follows reconstruction error much more strongly: reducing
residual normalized MSE from 12.47% to 4.34% and 1.68% progressively reduces
loss from 4.3323 to 3.2882 and 2.6613.

The three-plane result improves the previous best one-code residual loss of
5.098914 by **2.437582**, but it is still 0.5487 behind the 2.1126 COAT-A4
control and uses six physical activation bits.

The exact-two-bit survivor has now completed 5,000 additional steps. It reached
**4.258285** loss (perplexity **70.6887**), an improvement of 0.074021 over its
short-run checkpoint. Only 0.005699 of that improvement occurred during the
last 2,000 steps. This is a measured optimization plateau: more training with
the same representation is not a credible route from 4.26 to the A4 range.

### Residual-plane survivor audit

We then evaluated every survivor over all 4,907,776 available non-overlapping
validation targets. These are the authoritative numbers; the earlier table is
the matched 100-batch screen used to select runs.

| Survivor | Plane allocation | Physical bits / scalar | Full validation loss | Perplexity |
|---|---|---:|---:|---:|
| Exact binary | 2 per layer | **2** | 4.251708 | 70.2252 |
| Exact binary | 3 per layer | 3 | 3.750950 | 42.5615 |
| Ternary, short | 3 per layer | 6 | 2.651519 | 14.1756 |
| Ternary, extended | 3 per layer | 6 | 2.598547 | 13.4442 |
| Ternary, late-layer mix | `[2,2,2,3,3,3]` | 5 average | 2.889333 | 17.9813 |
| Ternary, NMSE-aware mix | `[3,2,3,2,2,3]` | 5 average | **2.858985** | **17.4438** |
| Ternary, ReLU-hardened | 3 per layer | 6 | **2.518279** | **12.4072** |

The exact three-bit binary arm proves that another sign plane helps, but it
still trails the six-bit ternary representation by 1.15 loss. At equal
five-bit average storage, assigning extra planes to layers with the highest
measured reconstruction error beats assigning them to the final three layers
by 0.03035 loss. Most importantly, replacing GELU with ReLU improves the
extended three-plane checkpoint by 0.08027. The integer-friendly nonlinearity
is therefore a quality win in this experiment, not a concession.

All seven checkpoints, exhaustive evaluations, diagnostics, resolved
configurations, and fixed-prompt generations are mirrored under
`residual-refinement-pilot/` on Hugging Face. The 27.4M-parameter float capacity
control is now running over the full 488M-token training corpus.

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

### Full GPU ternary-QKV result

Every selected arm then received the same 750-step / 6,144,000-token QAT budget
and was evaluated over 100 deterministic validation batches:

| QAT arm | Loss | Perplexity | Gate state |
|---|---:|---:|---|
| Ternary Q/K/V | **2.289840** | **9.8734** | — |
| Q-ViT rectification + binary gate | 2.561854 | 12.9598 | 100% open |
| Rectification + gate + structured distillation | 2.500275 | 12.1858 | 100% open |
| 2-bit score + integer LUT, no rectification | 2.301631 | 9.9905 | — |
| Same strict route + structured distillation | **2.299876** | **9.9729** | — |

Plain ternary Q/K/V passed the predeclared loss-below-2.50 screen. The strict
integer-LUT route cost only 0.010036 additional loss relative to that model, but
remains 0.160816 behind the matched A4-QKV/2-bit-probability control at 2.139060.
That is a successful architecture screen, not yet “same loss.”

Q-ViT-style rectification was counterproductive when introduced abruptly, and
the binary gates never closed. Distillation recovered 0.061579 loss in that
damaged arm, but only 0.001755 in the already stable no-rectification arm.
Fixed-prompt generations remain recognizably story-like; the unedited samples
are in
[`GATED_ATTENTION_GENERATION_SAMPLES.md`](GATED_ATTENTION_GENERATION_SAMPLES.md).

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

Appendix B shows how its attention kernels stay code-only: learned layerwise
scales \(s_Qs_K\) are applied after ternary Q·K, and \(s_{Att}s_V\) after
boolean-by-ternary Route·V. Our exact Q·K reference has the same factorization,
although it uses finer per-vector scales. Our current per-token V scales cannot
all be moved outside Route·V, so a learned shared V scale is now an explicit
ASIC-hardening experiment rather than an unverified assumption.

That ablation is implemented as `qkv_scale_granularity = "learned_head"`.
It learns one positive scale for each Q/K/V head, uses exact ternary forward
codes, and passes STE gradients to both the shadow activation and scale. After
the attention-clip refinement, a matched stage will screen initial scales 0.25,
0.5, 0.75, and 1.0 on the same 200 batches. It will then adapt the winning
learned-head scale and an unchanged per-token-scale control for 4,000 steps
each, followed by exhaustive validation of both. This isolates the quality
price of making the scale factorizable outside both attention matmuls from the
benefit of receiving more optimizer steps.

A one-step integration smoke loaded the real 27.4M ternary-weight checkpoint
through this new path, trained, validated, saved, reloaded, and generated text
on CPU. It added exactly 192 parameters—three scales × eight heads × eight
layers. Every scale and gradient remained finite, and clip 2 exercised all four
route codes. The immediate one-step loss is not a quality result: converting an
unadapted per-token-scale checkpoint abruptly is intentionally harsh, which is
why the queued experiment screens initialization and performs matched QAT.

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

### Other recent architecture evidence

[BinaryAttention](https://arxiv.org/abs/2603.09582), accepted at CVPR 2026,
reduces Q and K to their signs and executes Q·K with bitwise operations. Its
QAT, self-distillation, and learned score bias recover quality on vision and
diffusion Transformers, with a reported speed above 2x
FlashAttention2 on A100. This is direct evidence that a binary-Q/K branch can
preserve useful similarity geometry. It is not evidence for ternary V,
Route·V, residual boundaries, or causal language modeling: the paper actually
uses eight-bit attention coefficients and eight-bit V. If the current
ternary-QKV clip and shared-scale arms cannot close the attention gap, the
predeclared next fallback is a matched binary-Q/K, ternary-V arm using
Q/K relational distillation. Binary operands remain valid ternary-hardware
operands because they use the strict subset `{-1, +1}`.

The fallback is implemented as `binary_qk_ternary_v`, including dynamic and
learned-head-scale paths, diagnostic alphabet checks, deployment-contract
support, and a matched remote experiment. After the four primary refinements,
the selected endpoint seeds a 4,000-step ternary-QKV control and a 4,000-step
binary-Q/K arm. Both use the same Q/K-similarity distillation objective; only
the Q/K alphabet differs. The strict arm omits the paper's optional
dense/context bias because it would create a higher-precision bypass around
our low-bit score representation.

A two-step end-to-end integration smoke has also passed through initialization
from a checkpoint, teacher capture, logit/attention/QK/hidden distillation,
backpropagation, validation diagnostics, best-checkpoint selection, and final
checkpoint serialization. It observed exactly zero zero-codes in Q and K and a
ternary V alphabet. Its tiny random-data loss is deliberately not treated as a
quality result.

The live 27.4M strict run revealed that clip 3 leaves probability code 2 unused:
the integer LUT ratios map to probability codes `[0, 0, 1, 3]`, and the first
diagnostic measured 79.62% zero routes. A follow-up clip-utilization refinement
is therefore queued after the strict baseline:

1. evaluate clips 1.5, 2.0, 2.5, and 3.0 on the same checkpoint and 200 batches;
2. adapt clips 1.5, 2.0, and 2.5 for 2,000 matched steps each;
3. select the lowest matched validation loss, including the mature clip-3
   source as a rejection control;
4. refine the winner for 5,000 more steps;
5. run exhaustive sequential validation, diagnostics, and fixed generations.

Clip 2 has a specific representational advantage: its ideal LUT ratios reach
all four probability codes `[0, 1, 2, 3]`. The sweep still allows the data to
reject that theoretical advantage if a different clip produces lower loss.

The longer clip-3 control completed its predeclared training budget. Its
matched validation trajectory through step 30,000 is:

| Step | Validation loss | Perplexity | Zero routes | Attention entropy | Residual NMSE |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 2.418388 | 11.2278 | 79.62% | 2.3844 | 0.01670 |
| 4,000 | 2.358272 | 10.5727 | 82.86% | 2.2176 | 0.01737 |
| 6,000 | 2.304905 | 10.0232 | 84.68% | 2.1034 | 0.01806 |
| 8,000 | 2.268195 | 9.6619 | 85.77% | 2.0259 | 0.01899 |
| 10,000 | 2.250819 | 9.4955 | 86.25% | 1.9860 | 0.01992 |
| 12,000 | **2.250422** | **9.4917** | 87.24% | 1.9178 | 0.02102 |
| 14,000 | 2.256556 | 9.5501 | 87.51% | 1.8916 | 0.02187 |
| 16,000 | 2.280490 | 9.7815 | 87.75% | 1.8672 | 0.02219 |
| 18,000 | 2.292726 | 9.9019 | 87.96% | 1.8620 | 0.02274 |
| 20,000 | 2.295187 | 9.9263 | 88.15% | 1.8451 | 0.02287 |
| 22,000 | 2.284187 | 9.8177 | 88.21% | 1.8340 | 0.02319 |
| 24,000 | 2.306608 | 10.0403 | 88.27% | 1.8314 | **0.02332** |
| 26,000 | 2.316209 | 10.1372 | **88.36%** | **1.8254** | 0.02327 |
| 28,000 | 2.306657 | 10.0408 | **88.42%** | **1.8170** | **0.02358** |
| 30,000 | 2.317175 | 10.1470 | **88.62%** | **1.8061** | 0.02355 |

The last 2,000 steps recovered only 0.000397 loss while route sparsity and
residual error continued to rise; the following three 2,000-step intervals then
regressed by 0.006134, 0.023934, and 0.012236 loss, followed by a smaller
0.002461 regression at 20k. Step 22k recovered 0.011000, but remained 0.033766
worse than the 12k minimum while route sparsity and residual error reached new
highs. Step 24k then worsened by 0.022421 and step 26k lost another 0.009601,
ending 0.065787 above the 12k minimum and erasing the prior interval's partial
recovery. Step 28k recovered 0.009552 loss, but remained 0.056235 above the
minimum. Route sparsity and residual NMSE reached new highs while attention
entropy fell again, so the loss recovery does not reverse the attention
collapse signal. Step 30k then worsened by 0.010518 loss and finished 0.066753
above the 12k minimum. Route sparsity reached 88.62%, entropy fell to 1.8061,
and probability code 2 remained unused. The geometry reached a measured
plateau followed by sustained degradation with two temporary recoveries.
Completing the predeclared budget prevents an adaptive stopping decision from
biasing the comparison. The rising zero-route fraction, falling entropy, and
permanently unused probability code 2 are concrete evidence for the
predeclared clip sweep; they were not used to alter the control mid-run.

The separate exhaustive sequential audit changes the absolute score slightly
but confirms the conclusion. Over 4,907,776 validation targets, the final 30k
checkpoint scores **2.303869 loss** and **10.01285 perplexity**. The preserved
10k checkpoint scores **2.240969 loss** and **9.40244 perplexity**, a 0.062900
improvement with no extra checkpoint selection. On the same full stream,
zero-route use is 87.93% versus 85.23%, attention entropy is 1.8873 versus
2.0798, and residual NMSE is 0.02430 versus 0.02068 for 30k versus 10k. This
confirms that prolonging the unchanged strict geometry degraded both quality
and its internal attention/residual diagnostics.

The running process was launched before best-checkpoint retention existed, so
its periodic `checkpoint.pt` would have overwritten the step-10,000 state at
15,000. We preserved that exact 337,312,923-byte checkpoint as
`checkpoint-step-10000.pt`; both files matched SHA-256
`d543268160bf9e5405a1b89747665fc1dca54686535cc0129bd8863e5d17a160`
before the overwrite. Both the 30k and preserved 10k checkpoints are now
separately exhaustive, generated, packed, checksummed, and artifact-audited.
Downstream arms save `best-checkpoint.pt` at every improved validation and
select the best available saved checkpoint between stages, while final
checkpoints remain the matched fixed-budget comparison.

[PT2-LLM](https://openreview.net/forum?id=7QZanjCD6M) adds activation-aware
ternary grid alignment and structural-similarity reordering for weights.
[From Attention to Activation](https://openreview.net/forum?id=IjduZQK8gM)
motivates an explicit no-update attention state, while
[hyperspherical 4-bit Transformers](https://openreview.net/forum?id=tiqfxkYf1o)
motivate bounded cosine-like scores. These are useful follow-up architecture
hypotheses, not evidence that a two-bit language-model attention path already
matches float loss.

The next architecture arm after the two queued refinements is therefore
`integer_lut_no_update`: add a fixed denominator bucket with no corresponding V
vector before score shifting and low-bit route normalization, matching the
Softmax-1 idea without introducing a floating operand. It will be screened from
the best shared-scale checkpoint against the unchanged integer-LUT path. The
gate is lower matched validation loss plus non-collapsed route-code utilization;
a theoretical outlier advantage is not enough.

The option is implemented as `attention_normalization = "softmax1"`, including
the virtual route in both integer-LUT and probability-code normalization. The
diagnostics report its realized no-update mass, and the exact Route·V reference
accepts the virtual two-bit code in its INT32 denominator. After the shared-
scale checkpoint exists, standard Softmax and Softmax-1 will start from that
same checkpoint, receive 4,000 steps each, and undergo exhaustive validation.
The reported delta therefore has a matched optimizer-budget control.

The selected attention checkpoint then enters one final nonlinear hardening
test. The current strict graph still contains GELU, whose tanh/polynomial
approximation is not a ternary operation. Earlier small-model evidence favored
ReLU-hardened three-plane residuals at loss 2.518279 versus 2.598547 for the
longer GELU arm. At 27.4M parameters, unchanged GELU and ReLU will therefore
start from the same selected attention checkpoint, receive 4,000 steps each,
and undergo exhaustive validation. ReLU removes a floating nonlinear
approximation from inference; it wins the endpoint only if its matched loss is
no worse.

After the matched binary-Q/K fallback, an inference-only normalization screen
now compares unchanged floating RMSNorm with the exact fixed-point
`integer_reference` runtime on the same selected checkpoint. It first runs a
matched 200-batch gate, then exhaustive sequential validation for both paths,
fixed generations, diagnostics, and a deployment export whose contract records
the chosen normalization arithmetic. The predeclared practical gate is no more
than +0.02 exhaustive validation loss. This PTQ screen receives no optimizer
steps, so it cannot confuse additional training with arithmetic equivalence;
QAT is required only if the exact path misses that gate.

The July 2026 literature audit also rejects three tempting but invalid shortcuts:

- TWLA's A4 is a mixed per-layer `{2,4,6,8}` budget, not uniform ternary
  activations;
- TurboAttention performs Q/K/V attention in INT8 and mixes INT2/INT4 KV heads;
- BWLA's best stable joint result is W1A6 and its low-rank residual correction
  is not allowed by our inference contract.

One older result is a direct positive precedent rather than a shortcut:

- [TBT](https://arxiv.org/abs/2306.01841) trained fully ternary BART/mBART
  (`2-2-2`) for summarization and translation. It uses
  `{-alpha, 0, +alpha}` for signed activations and
  `{0, alpha, 2*alpha}` for nonnegative Softmax/ReLU outputs. Its fully ternary
  quality remained below full precision and it does not report causal-LM loss
  or a complete integer runtime, but it validates the signed/nonnegative
  codebook split already present in our residual and attention routes. Its
  max-entropy weight lesson is also reflected in the running strict model's
  near-balanced ternary weight fractions.

Two positive hardware references still inform the implementation:

- [IntAttention](https://arxiv.org/abs/2511.21513) demonstrates an integer-only
  softmax path with a 32-entry lookup table and direct integer normalization.
  It validates the dataflow, but at INT8 rather than ternary operands.
- [ELiTeFormer](https://arxiv.org/abs/2607.03652) demonstrates ternary linear
  projections and hybrid linear attention on an FPGA. Its cache compression
  and reported quality make it a future architecture branch, not a result for
  the current causal-softmax model or uniform ternary activations.

- [PackQViT](https://openreview.net/forum?id=N56hAiQvot) supports a fully
  four-bit vision path with integer-friendly nonlinear approximations. It
  strengthens the implementation case, but does not establish ternary language
  modeling.
- [Bipolar Self-Attention](https://openreview.net/forum?id=nG45z7lJ7D) uses
  ternary Q·K scores and a shift-based softmax in spiking vision models. It is
  useful evidence for multiplier-free attention, at a different model and data
  regime.
- [CAT-Q](https://arxiv.org/abs/2606.26650) uses modulation and softened
  ternarization for post-training weight conversion. It may improve weight
  adaptation, but it is weight-only rather than a solution to low-bit
  activations or residuals.
- [FiX](https://openreview.net/forum?id=WsNpCXq6SG) argues that the attention
  denominator can be removed when an immediately downstream RMSNorm cancels its
  scale. Our current pre-norm residual block adds the attention branch before
  the next norm, so this equivalence does not directly apply. A sandwich-norm
  or branch-normalized architecture is a valid future arm.

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

### Gate 1: Recover ternary Q/K/V — passed

The equal-budget 750-step arms are complete. Plain ternary Q/K/V reached 2.289840
and the strict integer-LUT path reached 2.299876, both below the 2.50 screen.
This authorizes longer runs and repeated seeds, but neither is yet within 0.05
of the 2.139060 matched control.

### Gate 2: Repeat and refine structured distillation

Use the existing COAT A4 model as teacher. Optimize:

- next-token cross-entropy;
- temperature-scaled logit KL;
- attention-probability MSE, from BWTA;
- sampled Q-Q and K-K similarity MSE, from Q-ViT.

The first strict-route comparison changed loss by only 0.001755, so the effect
is not yet distinguishable from run variance. Repeat the two best arms for three
seeds before tuning distillation weights. Preserve distillation for unstable
architectural transitions, where the first pilot recovered 0.061579.

### Gate 3: Smooth multistage residual reduction — completed

The BWTA-inspired schedule was implemented and executed as
`19 → 15 → 11 → 9 → 7 → 5 → 3`. It is substantially better than an abrupt
ternary switch, but the loss curve exposes a sharp degradation below seven
levels. The final single-plane result is 5.098914 and therefore fails the
quality gate.

### Gate 4: Increase residual capacity without abandoning ternary operators

The first, second, and single-plane fifth items have now been tested. The next
best-controlled experiment is a **multi-plane ternary residual**:

1. represent each residual with two separately scaled ternary planes;
2. represent each residual with three separately scaled ternary planes;
3. constrain plane scales per channel to fixed-point or powers of two;
4. reconstruct each block against the A4 teacher before end-to-end QAT;
5. compare the result with matched INT3 and A4 controls.

This keeps every matrix operand ternary and permits an ASIC to compute each
plane with add/subtract/skip operations, but honestly uses more than one ternary
code per scalar. It is not the same 1.58-bit storage claim as the rejected
single-plane endpoint.

Alongside that main ladder, test four architecture-focused arms:

- blend Q-ViT rectification gradually from the identity instead of switching it
  on abruptly;
- train a soft no-op gate with a sparsity target, then harden it to binary for
  inference;
- replace the unused four-level route with a learned three-level or logarithmic
  codebook;
- test branch RMSNorm or sandwich normalization so the FiX denominator-removal
  condition can be evaluated without silently changing the current algebra.

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
