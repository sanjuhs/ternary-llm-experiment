# Can an Entire Language Model Think in Three Values?

## A running, evidence-first account of the Ternary LLM Experiment

*Last updated: July 25, 2026*

The appealing version of this idea is wonderfully simple: replace most numbers in
a language model with only `-1`, `0`, and `+1`. The model becomes dramatically
smaller, multiplication can often become add/subtract/skip, and specialized
hardware could become much simpler.

The difficult version of the question is more interesting:

> Can the model preserve useful language while its weights **and its persistent
> hidden state** are repeatedly reduced to three values?

That is what this project tests. It does not quietly call a model "ternary" when
only its stored weights are ternary and all of its changing internal state remains
16-bit floating point.

The short answer so far is:

- ternary weights work surprisingly well;
- 4-bit activations remain usable;
- forcing every important activation boundary to three values causes a large
  quality collapse;
- COAT's correlation-aware rotation is mathematically sound and improves
  quantization conditions, but a rotation alone does not recover information after
  an activation is crushed into only three levels.

That last result is not the end of the idea. It tells us which part needs a new
architecture or learning rule.

## What “fully ternary” can honestly mean

A neural network cannot use only three values everywhere. Even the dot product of
two ternary vectors needs a wider accumulator: a length-\(n\) dot product has
\(2n+1\) possible exact sums. Softmax, normalization statistics, logits, the loss,
and training gradients also need wider numerical ranges.

Our operational definition is therefore:

1. learned matrix operands are stored or presented as ternary codes;
2. persistent hidden states are stored as a ternary code plus a scale;
3. reductions use wider temporary accumulators;
4. the result is rescaled and requantized before becoming the next persistent
   state.

That is analogous to an integer inference engine. “Low-bit” describes stored
operands and boundary representations, not every temporary register.

Three possible symbols carry \(\log_2(3)\approx1.585\) bits of information in an
ideal entropy code. In a simple random-access implementation, four ternary weights
fit into one byte, so we call the physical format **2-bit packed ternary** and avoid
claiming an impossible 1.58-bit fixed-width memory cell.

## The controlled experiment

We trained a small decoder-only transformer on the complete TinyStories training
split:

| Item | Value |
|---|---:|
| Training stories | 2,119,719 |
| Training tokens | 488,174,163 |
| Validation tokens | 4,907,807 |
| Parameters | 5,836,032 |
| Layers / width / heads | 6 / 256 / 8 |
| Context length | 256 |
| Tokenizer vocabulary | 4,096 |
| Tokens sampled per full run | 488,177,664 |

The float and ternary-weight models saw the same token budget and used the same
architecture, tokenizer, seed, and evaluation procedure.

### Full-corpus result

| Forward representation | Validation loss | Perplexity |
|---|---:|---:|
| Floating point | 1.6989 | 5.47 |
| Ternary weights, float activations | 2.0312 | 7.62 |

This is the most encouraging result in the project. Replacing the learned weights
with scaled ternary values costs quality, but the model still learns TinyStories
well enough to generate recognizable story text.

The inference-only packed export is 1,521,077 bytes, compared with 70,102,491 bytes
for the resumable training checkpoint. The latter also contains floating shadow
weights and AdamW optimizer state, so this is a deployment-format comparison, not a
claim that training memory fell by the same factor.

## Where strict activation ternarization failed

A 1,000-step matched pilot showed a much sharper distinction:

| Mode | Validation loss | Perplexity |
|---|---:|---:|
| Float | 3.215 | 24.91 |
| Ternary weights | 3.572 | 35.58 |
| Ternary activations | 6.053 | 425.54 |
| Ternary weights and activations | 6.047 | 422.87 |

The diagnostic signature was progressive damage to the residual stream. In the
activation-only run, mean absolute residual magnitude grew from 3.88 after the
first block to 66.91 after the sixth. The fraction of features changing sign
between blocks fell from 0.414 to 0.070. The model was not merely a little noisier:
its state became large, sticky, and information-poor.

We tested two direct remedies:

- two and four deterministic ternary population lanes, which add representable
  levels;
- residual branch scaling, which limits magnitude growth.

Neither crossed the quality gate. Population coding controlled magnitude better
but did not restore a confident output distribution. Branch scaling stabilized
gradients but froze residual signs through the middle layers.

## What the COAT paper actually contributes

[COAT: COrrelation-Aware Orthogonal Transform for LLM
Quantization](https://openreview.net/forum?id=6JGbm48VoG) starts from a useful
invariance. For an orthogonal matrix \(R\):

\[
XW^\top = (XR)(WR)^\top.
\]

The model's unquantized function is unchanged, but the coordinate system is
different. A good rotation spreads activation energy across channels, reducing the
outliers that make low-bit rounding destructive.

COAT improves on a random Hadamard rotation. Given calibration covariance
\(\Sigma=U\Lambda U^\top\) and a normalized Hadamard matrix \(H\), it chooses:

\[
Q=UH.
\]

This makes the diagonal of \(Q^\top\Sigma Q\)—the variance seen by each rotated
coordinate—equal in the ideal calibration estimate. The paper applies several
rotations through a Transformer, calibrates its important \(R_1\) and \(R_2\)
rotations, and fuses them into adjacent weights.

Two qualifications matter:

1. COAT evaluates **W4A4/KV4**, not ternary activations.
2. It improves low-bit accuracy but does not generally reproduce float loss.

For example, the paper reports average float versus COAT perplexities of 16.88
versus 22.38 for Llama-2 7B, 20.85 versus 24.91 for Llama-2 13B, 11.09 versus
11.87 for Llama-2 70B, and 8.92 versus 11.49 for Llama-3 8B. The 70B result is
strong, but “the loss stays the same” would overstate the evidence.

## Our COAT-inspired extension

We implemented a deliberately scoped first extension:

- observe residual states from the trained ternary-weight checkpoint;
- estimate their covariance with a streaming Welford accumulator;
- compute the paper's closed-form \(Q=UH\);
- rotate the residual vector into that basis;
- quantize it to either 16 levels (A4 control) or three levels (ternary test);
- decode through \(Q^\top\) for ordinary PyTorch reference kernels.

The implementation passed three important numerical checks:

| Projection check | Result |
|---|---:|
| Calibration residual samples | 24,576 |
| Variance coefficient of variation before | 0.093699 |
| Variance coefficient of variation after | \(7.0\times10^{-9}\) |
| Maximum orthogonality error | \(7.2\times10^{-7}\) |

This confirms that the transform does what COAT says at the calibrated residual
boundary. It is not yet a full reproduction of every fused \(R_1\)–\(R_4\) path in
the paper, and the project labels it accordingly.

### Preliminary post-training activation result

These numbers use ten deterministic full-validation batches from the same
full-corpus ternary-weight checkpoint:

| Activation treatment | Loss | Perplexity |
|---|---:|---:|
| A16 baseline | 2.0251 | 7.58 |
| Hadamard A4 | 2.1712 | 8.77 |
| COAT-projected A4 | 2.1690 | 8.75 |
| Hadamard ternary | 8.6642 | 5,791.53 |
| COAT-projected ternary | 8.6724 | 5,839.59 |

The scientific interpretation is clearer than the raw numbers:

- the A4 control shows that a low-bit activation path can remain useful;
- the learned COAT basis slightly improves on a plain Hadamard basis at A4;
- both ternary versions collapse;
- therefore, equalizing channel variance is not sufficient when each vector
  component has only three representable values.

An orthogonal transform preserves information **before** quantization. It cannot
guarantee preservation **after** a many-to-one mapping to `-1`, `0`, or `+1`.

## Could quantization-aware training recover it?

Only partly. We initialized all four activation paths from the same full-corpus
ternary-weight checkpoint and fine-tuned each for 1,000 steps, or 16,384,000
sampled tokens. Training used the tokenizer-compatible 50,000-story shard and
evaluation used 100 deterministic batches from the full validation stream.

| Fine-tuned activation path | Validation loss | Perplexity |
|---|---:|---:|
| Hadamard A4 | 2.1156 | 8.29 |
| COAT-projected A4 | 2.1126 | 8.27 |
| Hadamard ternary | 6.3967 | 599.88 |
| COAT-projected ternary | 6.3798 | 589.82 |

The learned COAT basis was consistently, if modestly, better than Hadamard after
fine-tuning. A4 remained close to the ternary-weight/A16 checkpoint. The ternary
arms recovered from their catastrophic post-training loss of about 8.67 to about
6.38, but did not approach useful A4 quality.

The optimization trace is also a warning: initial ternary gradient norms were on
the order of \(10^{12}\), compared with less than one for A4. Gradient clipping
kept the runs finite, but the result is not “same loss.” This strengthens the case
that the limiting factor is representation and residual architecture, not merely a
missing calibration pass.

Promising additional approaches include:

- **teacher distillation:** train the ternary student to match float logits and
  intermediate states, not only the next-token label;
- **learned thresholds and scales:** replace the fixed `0.5` ternary threshold with
  per-layer parameters and regularize code balance;
- **block-wise reconstruction:** minimize each quantized block's output error before
  global language-model fine-tuning;
- **mixed precision for weak layers:** spend four bits only where sensitivity
  measurements show cascading error;
- **residual-free or quantization-friendly architecture:** eliminate the state path
  that repeatedly accumulates rounding error;
- **multiple ternary planes:** represent a value as a short sum of scaled ternary
  codes, accepting more than one code per logical scalar;
- **rotation-aware weight relocation:** jointly optimize the rotation and ternary
  weight centroids instead of rotating activations alone.

The trade-off is important: multiple ternary planes or selective A4 may reach the
same loss, but they are no longer a one-code-per-feature fully ternary network.
That can still be an excellent engineering result if it is named honestly.

## What the surrounding research says

[QuaRot](https://arxiv.org/abs/2404.00456) demonstrates end-to-end W4A4/KV4
inference with all matrix multiplications at four bits and no retained
high-precision outlier channels. Its Llama-2 70B result stays within 0.47
WikiText-2 perplexity of the unquantized model. This is strong evidence for
rotations at four bits, not at three activation levels.

[BitNet a4.8](https://arxiv.org/abs/2411.04965) pairs extreme low-bit weights with
4-bit inputs, 8-bit sparse intermediate states, and a 3-bit KV cache. Its mixed
design is further evidence that activation precision—not only weight
precision—is the hard boundary.

[TWLA](https://arxiv.org/abs/2606.13054) is the closest recent paper to this
project. It combines ternary weights with 4-bit activations, asymmetric ternary
weight relocation, Kronecker orthogonal shaping, and inter-layer-aware mixed
precision. It still targets W1.58A4, not ternary activations.

[The Quantization Benefits of Residual-Free
Transformers](https://arxiv.org/abs/2605.25880) directly supports our diagnostic:
residual mixing can drive activations away from Gaussianity and amplify
quantization error. The paper trades a small amount of full-precision quality for
much greater low-bit robustness using a residual-free architecture, orthogonal
initialization, specialized optimization, and depth-aware attention scaling.

[QuIP](https://arxiv.org/abs/2307.13304) and
[QuIP#](https://arxiv.org/abs/2402.04396) show how incoherence processing,
second-order-aware rounding, lattice codebooks, and fine-tuning make approximately
2-bit **weight-only** quantization viable. They are valuable baselines for storage,
but they do not solve ternary persistent activations.

## Can attention itself fit into two bits?

Attention contains several different numerical objects, so “2-bit attention”
needs a more exact definition:

1. Q and K are multiplied to produce attention scores.
2. A row maximum is subtracted for numerical stability.
3. softmax exponentiates and sums the shifted scores.
4. the normalized probabilities are multiplied by V.

The Q, K, and V operands can be ternary. The shifted score presented to the
exponential can use four codes. The normalized probability can also use four
codes—or even a binary keep/drop decision. The dot-product accumulator and
softmax denominator cannot be two bits because they sum many terms; they need a
wider integer representation before the output is requantized.

[EXAQ](https://openreview.net/forum?id=AuJ6gDjcZK) provides the most directly
relevant low-risk method. It subtracts the row maximum, clips the negative
softmax input, and quantizes it to as little as two bits. Four input codes permit
a four-entry exponential lookup table and cheaper grouped accumulation. EXAQ
reports baseline PIQA accuracy for LLaMA-1 30B and a 36.9% softmax acceleration.
It does **not** reduce the final normalized probability to two bits, so it solves
the exponential-input bottleneck rather than every attention boundary.

[I-LLM](https://arxiv.org/abs/2405.17849) shows a complementary path at W4A4.
Its integer softmax clips shifted inputs and implements the exponential with
integer shifts; integer normalization and dynamic integer matrix multiplication
complete the pipeline. This is evidence that all-float softmax is unnecessary,
but not evidence that uniform two-bit probabilities are sufficient.

[BWTA](https://arxiv.org/abs/2604.03957), published in April 2026, pushes
further. Its attention path uses ternary Q and K, a high-precision softmax,
binary attention probabilities with a learned scale, and ternary V. The custom
binary/ternary matrix-multiplication kernel supports both linear layers and
attention. Its smooth multi-stage quantizer progressively reduces the activation
alphabet and uses magnitude-alignment factors, intermediate-state distillation,
feed-forward-output distillation, logit distillation, and attention-probability
distillation.

BWTA is important, but its language-model table replaces only 30% of the
least-sensitive layers and keeps the first/last layers and nonlinear operations
at higher precision. Its average LLM representation is reported as W1.3/A6. It
therefore supports our proposed training mechanism without proving a fully
ternary LLM.

The paper's Appendix B also clarifies an important implementation detail.
BWTA uses learnable layerwise scales \(s_Q\) and \(s_K\) around its
ternary-by-ternary Q·K kernel, and learnable scales \(s_{Att}\) and \(s_V\)
around its boolean-by-ternary Route·V kernel. Because each scale is shared at
the layer level, the large matrix multiplication operates only on packed codes
and the scale product is applied afterward.

Our current reference is slightly different. Q and K use per-vector scales;
those remain factorizable after each integer Q·K dot product, and the repository
now tests that exact INT32 path. V also uses a per-token scale, however, and
different key positions are mixed by Route·V. Those V scales cannot all be
pulled outside the reduction as one scalar. A truly bit-serial Route·V kernel
therefore needs either a learned shared V scale (the BWTA route), fixed-point
scale multipliers inside the reduction, or scale-bucketed value planes. The
first option is the cleanest next ASIC-oriented ablation.

BWTA is also explicit that its LLM implementation retains global
Softmax/LayerNorm/GELU in BF16 and the first and last layers in full precision.
Its kernel evidence is strong evidence for the expensive matrix products, not
for our stricter whole-graph low-bit claim.

Two other findings explain why the remaining step is hard:

- [Q-ViT](https://arxiv.org/abs/2210.06707) found that quantizing Q,
  K, V, and attention weights was the most damaging part of a fully 2-bit
  vision Transformer, costing up to 10.03 percentage points in its ablation.
  It recovered accuracy with information rectification and
  distribution-guided attention distillation.
- [Quantizable Transformers](https://openreview.net/forum?id=sbusw6LD41)
  connects activation outliers to attention heads attempting to produce a
  no-update state. Clipped softmax or gated attention gives the architecture a
  cleaner way to do nothing and improves later quantization.

### Our matched 2-bit attention screen

We replaced PyTorch's opaque fused attention call with an explicit causal path
and implemented three fake-quantized boundaries:

- `score_int2`: shifted/clipped softmax inputs use codes
  `{-3, -2, -1, 0}`;
- `prob_int2`: normalized probabilities use codes `{0, 1, 2, 3}` with a
  per-row scale and are renormalized;
- `prob_binary`: routes above a fraction of the row maximum survive and are
  renormalized.

On ten deterministic full-validation batches from the same 1,000-step COAT-A4
checkpoint:

| Attention path | Validation loss | Perplexity | Loss change |
|---|---:|---:|---:|
| Float attention | 2.1063 | 8.22 | — |
| 2-bit softmax input | 2.1581 | 8.66 | +0.0518 |
| 2-bit probability | 2.1744 | 8.80 | +0.0681 |
| Binary probability, threshold 0.5 | 2.7772 | 16.07 | +0.6708 |

This is a screening result, not a final parity claim. It is nevertheless useful:
the two genuine four-code variants preserve most of the A4 checkpoint's quality
without any attention-specific fine-tuning. Binary routing needs a lower
threshold and/or the learned scale and distillation used by BWTA.

### Full 100-batch GPU result

The 100-batch screen confirmed the ordering and found a clear optimum for both
clip and binary threshold:

| Attention path | Setting | Loss | Perplexity | Change from control |
|---|---:|---:|---:|---:|
| Float attention | — | 2.1135 | 8.277 | — |
| 2-bit softmax input | clip 6 | 2.1671 | 8.733 | +0.0536 |
| 2-bit softmax input | **clip 8** | **2.1565** | **8.641** | **+0.0430** |
| 2-bit softmax input | clip 10 | 2.1754 | 8.806 | +0.0619 |
| 2-bit softmax input | clip 12 | 2.2043 | 9.064 | +0.0908 |
| 2-bit probability | — | 2.1800 | 8.846 | +0.0665 |
| 2-bit score + probability | clip 3 | 2.2402 | 9.396 | +0.1267 |
| 2-bit score + probability | clip 6 | 2.4360 | 11.427 | +0.3225 |
| Binary probability | threshold 0.0625 | 2.3161 | 10.136 | +0.2026 |
| Binary probability | **threshold 0.125** | **2.2259** | **9.262** | **+0.1124** |
| Binary probability | threshold 0.25 | 2.2670 | 9.651 | +0.1535 |
| Binary probability | threshold 0.5 | 2.7850 | 16.200 | +0.6715 |

Lower is not always better. A very low binary threshold retains too many routes
and makes the row too uniform; a high threshold discards too much context. The
two stacked four-code quantizers also need coordinated scales. At clip 6, the
score spacing makes the subsequent probability quantizer almost binary. Clip 3
preserves an intermediate nonzero level and substantially reduces the damage.

Every selected arm was then reinitialized from the same COAT-A4 checkpoint and
fine-tuned for 500 steps, or 8,192,000 sampled tokens:

| Fine-tuned attention path | Loss | Perplexity | Gap to trained control |
|---|---:|---:|---:|
| Float attention control | 2.1132 | 8.275 | — |
| 2-bit softmax input, clip 8 | 2.1489 | 8.576 | +0.0357 |
| **2-bit probability** | **2.1383** | **8.485** | **+0.0251** |
| 2-bit score + probability, clip 3 | 2.1569 | 8.644 | +0.0437 |
| Binary probability, threshold 0.0625 | 2.2085 | 9.102 | +0.0952 |
| Binary probability, threshold 0.125 | 2.1889 | 8.926 | +0.0757 |

The probability-only arm is the quality winner. Its perplexity is about 2.5%
higher than the matched control, so “identical loss” is not yet supported. The
strict combined arm is the more important systems result: both the softmax input
and normalized routing cross four-code boundaries, yet its loss is only 0.0437
above the trained control after a short adaptation.

The combined arm does not use all four probability codes: because the four
softmax-input values produce a structured exponential distribution, its
probability codes are predominantly 0, 1, and 3. A learned joint codebook or a
two-bit log/exponent representation is therefore a better next step than two
independent uniform quantizers.

Fixed-seed generations from three prompts remain recognizably TinyStories-like
for the control, best-quality two-bit arm, and strict combined arm. All three
show the grammar and repetition limitations expected from this 5.8M-parameter
model; none of the quantized arms exhibits a distinct collapse. The unedited
outputs are in
[the matched generation appendix](ATTENTION_GENERATION_SAMPLES.md).

## The second attention architecture: ternary Q/K/V by construction

The first attention experiment left one ambiguity: in `coat_a4` mode, Q, K, and V
inherited the four-bit activation format. The routing was two-bit, but the
operands that produced and consumed it were not ternary.

The new implementation removes that ambiguity. Q, K, and V have an independent
configuration and can be forced to scaled codes in `{-1, 0, +1}` regardless of
the residual-stream format. Evaluation now reports the negative, zero, and
positive code fraction for all three tensors at every layer.

This turns the next experiment into an evidence ladder:

1. keep the stable COAT A4 residual stream;
2. force only Q/K/V to ternary;
3. train learned Q/K rectification;
4. add a quantization-friendly binary attention gate;
5. replace the general exponential with a four-entry integer lookup table;
6. reduce the normalized route to two bits or one bit;
7. use distillation before attempting a ternary residual stream again.

### What comes from each paper

[BWTA](https://arxiv.org/abs/2604.03957) supplies the closest attention
dataflow—ternary Q/K/V, binary attention routes, smooth multistage degradation,
magnitude alignment, and multiple teacher targets. Its LLM evidence is still
partial: it quantizes the 30% least-sensitive layers rather than the entire
network.

[EXAQ](https://openreview.net/forum?id=AuJ6gDjcZK) motivates a two-bit
max-shifted softmax input and a four-entry exponential table. Our reference
stores Q15 integer numerators, adds an integer row denominator, and uses ordinary
softmax only as the straight-through backward surrogate during QAT.

[I-LLM](https://arxiv.org/abs/2405.17849) confirms that nonlinear layers do not
need to return to floating point between operations. Its DI-MatMul,
DI-ClippedSoftmax, shift-based exponential, and integer normalization demonstrate
an integer-only W4A4 pipeline. Its nonlinear activation target is eight bits, so
it is not proof of a fully ternary pipeline.

[Q-ViT](https://arxiv.org/abs/2210.06707) supplies the repair mechanism for the
most fragile area. Learned affine rectification follows per-head Q/K
standardization. The student can also match a teacher's sampled Q-Q and K-K
similarity matrices, preserving relational structure without demanding identical
ternary coordinates.

[Quantizable
Transformers](https://arxiv.org/abs/2306.12929) explains the gate. Some attention
heads create extreme logits because softmax gives them no clean way to produce a
no-update. A small per-head, per-token gate can nullify the attention branch
directly. Our inference gate uses ternary operands and emits a binary keep/zero
code; its sigmoid is a training surrogate and can become a fixed-point lookup on
an ASIC.

### A wider accumulator is not a float fallback

A length-32 ternary Q·K reduction has exact outputs from -32 to +32 and needs
about seven signed bits. The width-256 and width-1,024 reductions in this model
need about ten and twelve signed bits. INT32 is a convenient general-purpose
container, not the precision of the next activation.

The accumulator is followed immediately by a fixed-point scale, rounding, and a
ternary/INT4 write. Softmax similarly uses a wider integer denominator, and
RMSNorm uses a wider integer sum of squares. These temporary reductions are
compatible with a model whose large stored operands and boundary tensors remain
low-bit.

### The first forced-QKV screen

Post-training conversion is severe:

| Treatment on the existing 2-bit-route checkpoint | 10-batch loss | Perplexity |
|---|---:|---:|
| Ternary Q/K/V, otherwise unchanged | 3.5965 | 36.47 |
| Untrained Q-ViT rectification + ternary Q/K/V | 4.1656 | 64.43 |
| Rectified ternary Q/K/V + Q15 score lookup, clip 6 | 4.1938 | 66.28 |
| Same + two-bit route, clip 6 | 3.9111 | 49.95 |
| Same + binary route, clip 6 | 4.5143 | 91.31 |

Tightening the score clip from six to three made the integer table less steep.
Without rectification, the two-bit route result improved from the 3.9 range to
**3.3565**, and its route codes used three levels rather than collapsing to only
zero and three. Untrained rectification remained worse at 4.0143.

This rejects a simple checkpoint switch. It does not reject the architecture:
the Q-ViT parameters and binary gates have not learned anything yet. The
750-step GPU pilot therefore compares plain ternary-QKV QAT,
rectification-plus-gating QAT, and the full teacher-distilled integer-table arm.
The experiment is specified in
[`configs/gated_attention_pilot.toml`](../configs/gated_attention_pilot.toml) and
[`scripts/remote_gated_attention_pilot.sh`](../scripts/remote_gated_attention_pilot.sh).

### What QAT recovered

The completed 750-step RunPod pilot changed the conclusion materially:

| Equal-budget arm | Validation loss | Perplexity |
|---|---:|---:|
| Ternary Q/K/V | **2.289840** | **9.8734** |
| Q-ViT rectification + binary gate | 2.561854 | 12.9598 |
| Rectification + gate + structured distillation | 2.500275 | 12.1858 |
| Integer score LUT, no rectification | 2.301631 | 9.9905 |
| Integer score LUT + distillation, no rectification | **2.299876** | **9.9729** |

The result supports three narrower claims:

1. Ternary Q, K, and V are trainable in this model. The best arm passed the
   predeclared 2.50 loss gate and recovered most of the catastrophic PTQ drop.
2. A two-bit score, four-entry Q15 exponential lookup, integer denominator, and
   low-bit route add only 0.010036 loss beyond plain ternary Q/K/V in this run.
3. Abrupt Q-ViT rectification is not automatically beneficial. Its hard gates
   remained 100% open, while distillation recovered only part of its loss.

The matched A4-QKV/2-bit-probability control still scores 2.139060, leaving the
best strict arm 0.160816 behind. Therefore “ternary attention can generate” is
now supported; “same loss” is not. Repeated seeds and a longer corpus run are
the next quality test.

The strict route's nominal four-level codebook used only codes 0, 1, and 3; code
2 was never selected. That points toward a learned three-level or logarithmic
route rather than an assumption that four uniformly spaced codes are ideal.
Unedited generations are collected in
[`GATED_ATTENTION_GENERATION_SAMPLES.md`](GATED_ATTENTION_GENERATION_SAMPLES.md),
and the exact run inventory is in
[`EXPERIMENT_LEDGER_AND_ROADMAP.md`](EXPERIMENT_LEDGER_AND_ROADMAP.md).

## We finally crossed every residual boundary with three codes

The next run applied the gradual alphabet reduction that the earlier failure was
missing. Instead of jumping straight from a smooth activation to three values,
the student moved through 19, 15, 11, 9, 7, 5, and finally 3 symmetric values.
Each checkpoint initialized the next stage, and magnitude alignment prevented a
pure scale shock.

The strict endpoint also replaced GELU with ReLU and hardened attention to a
binary route. Its large persistent tensors are therefore ternary at inference:
weights, Q/K/V, residual boundaries, and feed-forward boundaries. Attention
scores cross a two-bit boundary and use a four-entry integer exponential lookup.
Dot products, normalization statistics, row sums, and scales still require
wider temporary state; that is compatible with a ternary data path and necessary
to avoid overflow.

| Residual codes available | Loss after adaptation |
|---:|---:|
| 19 | 2.3975 |
| 15 | 2.5101 |
| 11 | 2.7650 |
| 9 | 2.9931 |
| 7 | 3.3627 |
| 5 | 4.2366 |
| 3, GELU | 5.1947 |
| 3, ReLU + binary attention | 5.1424 |
| 3, plus 1,500 CE-only steps | **5.0989** |

This is both progress and a clear negative result. The old one-code residual
model scored 6.3798, so progressive training recovered 1.2809 loss. The final
model genuinely used all three residual codes and produced story-shaped
fragments. But useful grammar and coherence were lost, and it remains far behind
the 2.1126 A4 result.

The curve also tells us where the problem begins. Quality declines moderately
from 19 to 7 levels, then drops sharply at 5 and 3. That pattern points to
insufficient residual-channel capacity, not a broken attention implementation.
Another long run at three levels may shave the number down, but the last 1,500
steps improved only 0.0435.

The most informative next test is two- and three-plane ternary residuals. With
one shared scale, two planes give five summed levels and three give seven. Our
residual-refinement variant fits a separate scale to each plane, so it can form
up to \(3^p\) combinations from \(p\) ternary codes. Every matrix operand remains
ternary and an ASIC still uses add/subtract/skip operations, while the residual
stream retains more information. The tradeoff is honest: it is no longer one
1.58-bit code per scalar. We will compare those arms against matched INT3 and A4
controls and train each block to reconstruct its teacher before end-to-end
fine-tuning.

Unedited output is in
[`FULLY_TERNARY_GENERATION_SAMPLES.md`](FULLY_TERNARY_GENERATION_SAMPLES.md).

## Training plan for a strict low-bit Transformer

The results now support a staged plan rather than another direct A16-to-ternary
jump:

1. **Establish the matched controls.** Repeat A16, COAT-A4, and every attention
   variant on identical batches and at least three seeds.
2. **Make attention low-bit while residuals remain A4.** Select between 2-bit
   softmax inputs, 2-bit probabilities, their combination, and thresholded
   binary routing.
3. **Add teacher supervision.** Match the A16 teacher's logits, attention maps,
   attention outputs, and block outputs. Attention-map KL and attention-output
   MSE are better targets than raw score MSE because many score matrices produce
   similar normalized routing.
4. **Reduce the activation alphabet gradually.** Use odd level counts such as
   19, 15, 11, 7, and 3, align magnitude at every transition, and spend at least
   half the budget at the final ternary stage.
5. **Learn thresholds and scales per layer/head.** Fixed global thresholds are
   a diagnostic baseline, not a likely optimum.
6. **Reconstruct one block at a time.** Minimize each quantized block's output
   error before whole-model language training so early errors do not avalanche
   through the residual stream.
7. **Test a quantization-friendly architecture.** Gated/no-op attention and a
   residual-free control directly target the two mechanisms implicated by the
   current diagnostics and literature.
8. **Export and benchmark the real representation.** Pack ternary values into
   two bits, use low-bit operands with wide accumulators, requantize every
   persistent boundary, and compare memory, throughput, energy proxy, loss, and
   generation—not only checkpoint size.

The quality gates remain strict: a short run may eliminate a bad idea, but
“same loss” requires full-validation confidence intervals, repeated seeds,
fixed-prompt generations, and a matched float/A4 control.

## Current conclusion

Can COAT “ternarize the weights and generate everything in ternary” by itself?

**No.** The paper does not make that claim, its reported system is four-bit, and our
controlled extension shows a large gap between A4 and ternary activations even
after the covariance objective is solved almost perfectly.

Can the broader idea still work?

**Yes, in stages.** Ternary weights already work. W1.58A4 is strongly supported by
both our preliminary result and recent literature. A one-code ternary residual
stream likely requires a model designed around that information bottleneck, not a
post-training transform attached to an ordinary residual Transformer. The most
credible routes are quantization-aware training with distillation, joint
rotation/codebook optimization, and a residual-free architecture.

The project will keep all negative results. A useful experiment is not one that
always says yes; it is one that shows exactly which “yes” is supported.

## Three different finish lines

“Fully ternary” can hide three different engineering objectives. The experiments
now report them separately:

1. **Exact two-bit activation storage.** Two binary residual planes consume
   exactly two code bits per scalar. Binary is a subset of ternary arithmetic,
   so a ternary datapath can process both planes. Dynamic scales and wider
   accumulators are still required.
2. **Ternary matrix operands.** Several ternary planes may represent one logical
   activation. Every large matrix multiplication still consumes only
   `-1/0/+1` codes, but two planes occupy four physical bits per scalar and three
   planes occupy six unless entropy coding is used.
3. **Near-float quality.** Four-bit or layer-mixed activations are currently the
   strongest published route. They are completely quantized, but they are not
   uniformly two-bit.

Those goals overlap, but none implies the other. A design can have ternary
operators without two-bit storage, or two-bit storage without matching float
loss.

### What the newest papers add

[Binary and Ternary Natural Language
Generation](https://arxiv.org/abs/2306.01841), or TBT, is older than the papers
below but is the closest published generative precedent. It trained BART and
mBART with ternary weights and ternary activations for summarization and
translation. The fully ternary model reached ROUGE-L **29.07** on XSUM and
**38.30** on CNN/DailyMail, compared with **35.71** and **42.09** for the
full-precision BART control; fully ternary mBART reached **21.70 BLEU** on
WMT16 En-Ro versus **26.82** at full precision. This is real text-generation
evidence, but it is encoder-decoder fine-tuning on downstream metrics rather
than causal language-model pretraining or matched TinyStories loss.

TBT contributes two mechanisms directly relevant to this project. First, its
statistics-based isometric weight quantizer aims to use all three weight codes
nearly uniformly while preserving scale. Our strict checkpoint already has
approximately balanced `-1`, `0`, and `+1` weight fractions, so this condition
is largely satisfied rather than a missing rescue. Second, its elastic
activation quantizer distinguishes signed and nonnegative tensors:
`{-alpha, 0, +alpha}` for signed residual-like values, but
`{0, alpha, 2*alpha}` for Softmax and ReLU outputs. That is an important
architectural clarification. Our signed ternary residual planes and nonnegative
integer attention codebook already implement the same division of labor; the
matched ReLU arm will test whether the remaining feed-forward activation can
benefit from it. TBT does not document an end-to-end causal-LM integer
accumulator/requantization contract, and its quality gaps mean it does not prove
loss parity, but it rules out the claim that ternary activations cannot generate
coherent text at all. The authors' released
[training code](https://github.com/facebookresearch/Ternary_Binary_Transformer)
also confirms the fully ternary `2-2-2` configurations.

[BitNet v2](https://arxiv.org/abs/2504.18415) shows that online Hadamard mixing
can stabilize W1.58A4 training where the corresponding unrotated A4 treatment
diverges. It supports our fixed-Hadamard arm, but it does not establish A2.

[QuEST](https://arxiv.org/abs/2502.05003), published at ICML 2025, is the
strongest direct evidence that very-low-bit **forward operands** can still be
trained stably. Its revised study covers Llama-family models from 30M to 1.6B
parameters with
weights and activations from one to four bits. Its two transferable ideas are
Hadamard normalization followed by MSE-optimal code fitting, and a “trust”
gradient estimator that suppresses updates where forward quantization error
makes the surrogate gradient unreliable. Specifically, it masks a quantized
entry's backward gradient when its absolute reconstruction error exceeds one
quantization half-interval, and applies that mask in the Hadamard domain before
the inverse transform. The backward multiplications themselves remain
standard-precision. The paper's Pareto result favors W4A4, not W1A1, and
“weights and activations” does not mean that Softmax,
normalization, accumulation, requantization, and sampling are all ternary.
Therefore it does not satisfy our strict endpoint by itself. It does motivate
a matched next ablation: retain our ternary forward codebooks and add a
QuEST-style error-aware gradient mask, comparing it with the existing STE under
the same checkpoint, token budget, and exhaustive validation set.

[R2Q](https://arxiv.org/abs/2511.21736) decomposes a two-bit weight into two
successive binary residual kernels. Our exact-two-bit activation arm adapts that
representation to residual states. That activation use is our hypothesis, not a
result claimed by R2Q.

[TWLA](https://arxiv.org/abs/2606.13054) combines asymmetric ternary relocation,
Kronecker orthogonal shaping, and adjacent-layer-aware mixed precision. Its
strongest lesson for this project is that later or sensitive blocks should
receive more activation capacity. It still targets W1.58A4 rather than a
uniformly ternary residual stream.

The full ICML 2026 table makes that boundary precise. On Qwen3-32B, TWLA moves
WikiText-2 perplexity from **7.61** at FP16 to **8.94** with ternary weights and
FP16 activations, and to **9.71** at its A4 budget. That A4 result is marked
mixed precision: its dynamic program chooses per-layer activation precisions
from `{2, 4, 6, 8}` while meeting an average budget. It is powerful evidence
for rotation plus cross-layer sensitivity allocation, but it does not show that
every activation boundary can use one ternary code.

[BWLA](https://arxiv.org/abs/2605.00422) is an even sharper negative control.
Its Orthogonal-Kronecker Transform smooths activation tails, and a small
low-rank correction recovers binary-weight error. It obtains useful W1A6
results, including **11.92** WikiText-2 perplexity on Qwen3-32B versus **7.61**
at FP16. At A4, however, the paper reports **39.88**, **55.12**, and **34.77**
perplexity for LLaMA2-7B, LLaMA3-8B, and Qwen3-14B. The distribution shaping is
worth borrowing, but its low-rank floating correction would violate our
no-bypass inference rule and its A4 evidence does not support a uniform two-bit
endpoint.

[TurboAttention](https://arxiv.org/abs/2412.08585) proves that an attention
kernel can avoid FP32 softmax and execute its matrix multiplications as
integers. Its implementation first quantizes Q/K/V blocks to INT8, stores half
the KV heads at INT2 and half at INT4, and approximates the exponential with a
small lookup table plus a cubic polynomial. The paper explicitly reports
quality degradation for pure INT2 and keeps the rest of the Transformer in
FP16. Its integer dataflow supports our kernel design, while its precision
choices do not satisfy the all-ternary contract.

[TurboBoA](https://arxiv.org/abs/2602.04929), published at ICLR 2026, attacks a
different failure mode: quantizing one weight matrix while pretending the
other projections in the attention block are independent. It jointly
quantizes output channels, compensates error propagated by earlier quantized
layers, and refines the quantization grid with coordinate descent. That is a
useful calibration idea for a future post-training ternarization control, but
it remains a weight/PTQ method; it does not make Q/K/V activations, Softmax,
normalization, or residual boundaries ternary. Our matched weight-only result
is already much closer to its float teacher than the activation-constrained
models are, so the present experiment correctly prioritizes activation and
arithmetic-boundary training over another weight-only optimizer.

[IntAttention](https://arxiv.org/abs/2511.21513) removes the remaining
dequantize-softmax-requantize detour with a 32-entry exponential lookup table
and direct integer normalization. It reports up to 3.7x attention speedup and
61% energy reduction on Arm CPUs, but its operands are INT8 and the current
paper says code will be released later. This is strong evidence for our
integer-LUT denominator and requantization boundary, not evidence that ternary
Q/K/V alone retain language-model quality.

[BinaryAttention](https://arxiv.org/abs/2603.09582), accepted at CVPR 2026,
shows a more aggressive but narrower result: Q and K retain only their signs,
so Q·K becomes a bitwise one-bit operation. Quantization-aware training,
self-distillation, and a learned score bias recover much of the lost
similarity structure; the authors report more than
2x the speed of FlashAttention2 on an A100 and matched or improved accuracy on
their vision and diffusion benchmarks. The full method makes the precision
boundary clearer: its attention coefficients and V use eight-bit integers, and
its optional corrective bias may be a dense, position-sensitive, or
context-aware higher-precision term. Binary values are a subset of ternary
values, so this is a legitimate fallback for our Q/K operands. It does **not**
establish ternary V, two-bit Route·V, low-bit residual boundaries, or
autoregressive language-model loss. The transferable experiment is therefore
a matched `binary Q/K + ternary V` arm with Q/K relational distillation—not a
claim that the paper has solved our whole inference graph.

That fallback is now implemented as
`qkv_quantization = "binary_qk_ternary_v"`. Dynamic per-token scales and
factorizable learned per-head scales both preserve exact binary Q/K and ternary
V code alphabets, with STE gradients for QAT. After the clip, shared-scale,
Softmax-1, and ReLU gates complete, a matched experiment will train an
unchanged ternary-QKV control and the binary-Q/K arm for 4,000 steps each from
the same selected checkpoint. Both receive the same Q/K-similarity
distillation weight, and selection uses exhaustive sequential validation.
A two-step integration smoke has already exercised the complete checkpoint,
teacher-distillation, backward, validation, diagnostic, and serialization
path. It confirmed exact binary Q/K alphabets and ternary V; its random-data
loss is not a quality measurement.

The remaining RMSNorm boundary is now executable rather than merely described.
An opt-in `integer_reference` path converts each incoming norm vector to signed
fixed point, accumulates squares in INT64, computes a tensorized exact integer
square root and integer division, and multiplies by ternary norm-weight codes.
Its forward pass is the integer reference; its backward pass uses the ordinary
RMSNorm derivative as a straight-through surrogate. On the same local smoke
checkpoint and batch, floating RMSNorm measured loss 6.19863 and the integer
path 6.19745. That tiny random-data comparison establishes wiring only. The
27.4M endpoint still needs a matched PTQ screen and, if necessary, QAT before
normalization can be called quality-preserving.

We deliberately omit BinaryAttention's optional dense/context bias: an
unbounded floating bias matrix would be a hidden bypass around the low-bit
score path. Our dynamic-scale reference uses per-vector magnitude alignment;
the stricter queued path instead learns one positive scale per head so the
scale can be applied outside the binary dot product. This is a contract-driven
adaptation of the paper, not an exact reproduction. In particular, our
explicit Q/K-similarity loss comes from the already implemented Q-ViT-style
distillation path; BinaryAttention reports ordinary teacher self-distillation
and observes sign-aligned similarity rather than defining that same loss.

[ELiTeFormer](https://arxiv.org/abs/2607.03652) is the closest July 2026
hardware proof: it combines hybrid linear attention, ternary linear
projections, and an FPGA processing element that replaces ternary
multiplication with bitmask operations. The authors report 10x weight and
12.8x KV-cache compression, with 31.9% MMLU—within three percentage points of
their BitNet b1.58 comparison. Its compressed cache and linear-attention state
are not claimed to be ternary activations, however. It supports a future
architecture branch and custom-ASIC feasibility; it cannot be counted as a
matched all-ternary autoregressive result.

[TeLLMe](https://arxiv.org/abs/2504.16266) is a complementary autoregressive
hardware result. It implements both prefill and decoding on an edge FPGA with
ternary weights, lookup-based matrix processing, fused attention, and an
integer-oriented normalization/quantization unit. Its persistent activations
are eight-bit, however. TeLLMe therefore proves that ternary-weight generation
can be engineered end to end on small hardware; it does not demonstrate our
ternary-Q/K/V, two-bit-route, or ternary-residual contract. This distinction
also explains why our wider integer accumulators and small fixed-point scales
are hardware-realistic without making the stored activation tensors FP32.

[From Attention to Activation](https://arxiv.org/abs/2410.17174) offers a
training-time route rather than a post-training codebook. Across its GPT-2
models, Softmax-1 and OrthoAdam keep unquantized perplexity essentially
unchanged while reducing hidden-state kurtosis toward three. On its 130M GPT-2
example, the 4-bit weight penalty falls from **657.0** perplexity points to
**1.2**, and the coarse 8-bit weight/activation penalty falls from **23.60** to
**0.43**. Embeddings, normalization, and softmax remain unquantized in those
experiments, so this is not an endpoint result. It is strong evidence that the
forward and optimizer geometry should prevent outliers before asking a tiny
codebook to represent them.

[FTerViT](https://arxiv.org/abs/2605.21171) closes another useful boundary by
ternarizing every weight matrix and normalization parameter in a vision
Transformer. Its activations remain eight-bit, and it is not a language model,
but TernaryLayerNorm is a concrete candidate for removing one of our remaining
learned full-precision parameter classes.

[RobuQ](https://arxiv.org/abs/2509.23582), revised in May 2026 and accepted by
ICML 2026, supplies complementary evidence at the activation boundary. It
reports ternary weights with average two-bit activations in diffusion
Transformers by using a Hadamard transform to regularize per-token
distributions and activation-only mixed precision to protect sensitive layers.
It is not an autoregressive language model and does not prove ternary attention
or residual operands. Its enhanced baseline also contains a floating low-rank
compensation branch, while its mixed-precision design fixes attention scores at
eight bits and adaptive LayerNorm at four bits. Average A2 therefore does not
mean an end-to-end two-bit graph. Its transferable claims are narrower:
orthogonal mixing can make a tiny activation codebook more stable, and
layerwise error should control where scarce extra bits are spent. Those are
exactly the two hypotheses tested by our fixed-Hadamard path and equal-storage
NMSE-aware plane allocation, without importing RobuQ's floating bypass.

[PT2-LLM](https://openreview.net/forum?id=7QZanjCD6M) is a complementary
weight-only result. Its iterative ternary fitting, activation-aware grid
alignment, and structural-similarity column reordering improve post-training
ternarization without retraining. It strengthens the case for calibration-aware
ternary centroids, but it does not quantize the persistent activation or
attention boundaries and therefore cannot by itself establish our inference
contract.

[From Attention to Activation](https://openreview.net/forum?id=IjduZQK8gM)
connects attention sinks and heavy-tailed hidden activations, and reports that a
softmax-minus-one formulation plus an optimizer change sharply reduces both.
This matters because our integer probability route already has an explicit zero
code: a head can represent “no update” without manufacturing an extreme logit.
The transferable hypothesis is architectural—make the zero-update state
explicit before quantization—not that the paper has already demonstrated
two-bit autoregressive attention.

For our integer-LUT attention, the closest equivalent is an explicit
**no-update bucket**: append a virtual zero-score key with no corresponding V
vector before score shifting, LUT lookup, and probability-code normalization.
Its code contributes to the integer denominator but not to Route·V. This is
algebraically analogous to Softmax-1 and is compatible with integer
accumulation. It is now a predeclared architecture arm after the clip and
shared-scale experiments, not an unmeasured change to the running control.
The implementation reports the realized no-update mass and supplies an exact
integer Route·V reference whose denominator includes the virtual route code.
The queued test now trains both the unchanged integer-Softmax path and
Softmax-1 for 4,000 steps from the same checkpoint. A pre-adaptation comparison
and exhaustive post-adaptation comparison prevent extra training time from
being mistaken for an architectural improvement. The shared-head-scale stage
uses the same rule: its learned-scale arm and per-token-scale control receive
equal optimizer budgets.

One additional boundary remains avoidably floating: GELU. ReLU needs only a
sign comparison and zeroing operation, so it is a cleaner custom-ASIC target.
Our earlier small-model three-plane experiment improved from 2.598547 with the
longer GELU arm to 2.518279 after ReLU hardening. The final queued ablation now
trains equal-budget GELU and ReLU arms from whichever integer-attention
normalization wins exhaustive validation. This is a test, not an assumption:
the ReLU endpoint is selected only if it preserves or improves matched loss.

The completed 27.4M strict control supports that ordering. At 2k, 4k, 6k, 8k,
10k, 12k, 14k, 16k, 18k, 20k, 22k, 24k, 26k, 28k, and 30k steps its matched
validation loss moved from
**2.41839** to
**2.35827**, **2.30490**, **2.26820**, **2.25082**, and **2.25042**
before regressing to **2.25656** at 14k and **2.28049** (perplexity **9.78147**)
at 16k, **2.29273** (perplexity **9.90189**) at 18k, and **2.29519**
(perplexity **9.92630**) at 20k, then recovering modestly to **2.28419**
(perplexity **9.81771**) at 22k before worsening to **2.30661**
(perplexity **10.04031**) at 24k and **2.31621** (perplexity **10.13717**)
at 26k, recovering to **2.30666** (perplexity **10.04080**) at 28k, and
finishing at **2.31717** (perplexity **10.14697**) at 30k. The
10k-to-12k interval improved loss by only 0.00040, then the next three
intervals lost 0.00613, 0.02393, and 0.01224; the fourth lost another 0.00246.
The 20k-to-22k interval recovered 0.01100, but remained 0.03377 worse than the
12k minimum; the next two intervals lost 0.02242 and 0.00960 and ended 0.06579
above that minimum. Step 28k recovered 0.00955, but remained 0.05623 above the
minimum. Over the full trajectory the zero-route fraction rose from 79.62% to
88.62%, attention entropy fell from 2.3844 to 1.8061, residual NMSE rose from
0.01670 to 0.02355, and probability code 2 remained unused. The loss recovery
at 28k therefore did not reverse the internal route-collapse trend; 30k lost
another 0.01052 and ended 0.06675 above the 12k minimum. This is a measured
plateau followed by sustained degradation with two temporary recoveries,
rather than a monotonic convergence curve. Completing the predeclared 30k
control keeps its endpoint unbiased. The diagnostic trend is the reason to run
the already-declared clip sweep afterward; it was not used to alter the
control midstream.

The full sequential audit covers 4,907,776 validation targets. It scores the
30k endpoint at **2.303869 loss** and **10.01285 perplexity**, while the exact
preserved 10k checkpoint reaches **2.240969 loss** and **9.40244 perplexity**.
That 0.062900 advantage confirms the bounded-validation trend. On the same full
stream, 30k versus 10k has 87.93% versus 85.23% zero routes, entropy 1.8873
versus 2.0798, and residual NMSE 0.02430 versus 0.02068. Both checkpoints have
now produced fixed generations, packed 2-bit-weight exports, complete
checksums, and successful artifact audits. The samples are recognizably
TinyStories-like but still contain repetitions and local contradictions, so
readability does not override the loss gap.

[TQL](https://openreview.net/forum?id=lwHSE1xYHH), published at ICML 2026,
independently identifies attention-entropy collapse as a cause of unstable
Transformer training and reports that explicitly controlling attention entropy
stabilizes its larger value-function models. It is an RL result, not a
quantized language-model result, so it does not validate our endpoint by
itself. It does strengthen the diagnosis suggested by our simultaneous
zero-route growth and entropy decline. If clip utilization, shared QKV scales,
and Softmax-1 do not recover enough loss, the next clean experiment is an
equal-budget QAT arm with an entropy-floor penalty against the float teacher,
selected solely by exhaustive language-model validation. That arm must still
use the same ternary/low-bit inference graph; the entropy term is training-only
guidance, not a floating inference bypass.

The live trainer predates automatic best-checkpoint retention and writes its
resumable checkpoint every 5,000 steps. Before the next overwrite, we therefore
preserved the exact step-10,000 checkpoint—the best checkpoint actually written
before the 12k metric minimum—as `checkpoint-step-10000.pt`, with SHA-256
`d543268160bf9e5405a1b89747665fc1dca54686535cc0129bd8863e5d17a160`.
The finalizer evaluated, diagnosed, generated from, exported, and audited that
checkpoint as a separate preserved run. The clip experiment starts from it
while still reporting the full 30k control. All newly launched training arms
retain `best-checkpoint.pt` automatically, and every subsequent stage records
which checkpoint it used.

[Accurate 4-Bit Quantization with Hyperspherical
Architecture](https://openreview.net/forum?id=tiqfxkYf1o) bounds attention and
MLP error growth by normalizing activations and constraining weights so that
dot products behave like bounded cosine similarities. Its evidence is at W4A4,
but the bounded-score principle is directly testable in a future ternary arm
and may reduce how much clipping the two-bit score codebook must absorb.

[CAT-Q](https://arxiv.org/abs/2606.26650) improves post-training ternary weights
with learnable modulation, softened ternarization, and sliding-layer
reconstruction. Those ideas are useful for the weight-conversion stage, but the
paper retains A8/A16 activations and therefore does not solve our residual
bottleneck.

[ExTernD](https://arxiv.org/abs/2607.13511) takes a different route: expand a
weight matrix into two ternary factors around a diagonal scale. Its reported
Qwen3.5-4B result at expansion factor three is close to BF16 perplexity, but the
effective storage is about 5.7 bits per original weight and the expanded factors
increase operation count. It is a credible ternary-compute branch, not an exact
two-bit-storage branch.

[The Quantization Benefits of Residual-Free
Transformers](https://arxiv.org/abs/2605.25880) attacks the activation
distribution itself. Removing additive residual accumulation keeps activations
closer to Gaussian and easier to quantize. Its evidence is at higher activation
precision, so a residual-free TinyStories model is an architecture experiment,
not yet proof of a ternary endpoint.

[TernaryLM](https://arxiv.org/abs/2602.07374) is a useful same-dataset caution.
It trains all projection **weights** ternarily from scratch with adaptive
layer-wise scales, but retains ordinary activations and reports TinyStories
validation perplexity 58.42 after training on only 12.9M tokens. Its tokenizer
and validation protocol differ from ours, so the raw perplexity is not a
baseline target. Its relevant lessons are native quantization-aware training,
learned layer scales, and greater sensitivity in boundary layers—not evidence
that ternary residual activations already match float quality.

### A live codebook-utilization finding

The first 27.4M strict checkpoint exposed a subtle problem that a headline
“two-bit” label hides. With attention clip 3, the integer exponential LUT is
approximately `[exp(-3), exp(-2), exp(-1), 1]`. The following row-wise
four-code probability quantizer maps those ratios to `[0, 0, 1, 3]`; probability
code 2 is therefore unused. The measured step-2,000 diagnostic confirmed exactly
that collapse and found 79.62% zero probability codes.

This is still a two-bit storage field, but it is functionally a sparse
three-level router. Clip 2 instead maps the ideal ratios to `[0, 1, 2, 3]`,
making every code reachable. The new matched refinement stage screens clips
1.5, 2.0, and 2.5, adapts each for the same token budget, compares them with the
already-mature clip-3 checkpoint on the same held-out batches, and exhaustively
evaluates a longer refinement of the winner.
It is a targeted test of codebook utilization, not an after-the-fact change to
the already-running clip-3 baseline.

That screen is now measured. On the same 200 validation batches, clip 1.5
improved from 4.392865 to 3.207251 after 2,000 QAT steps but made the zero code
completely unreachable. Clip 2.5 improved from 2.336622 to 2.227716, yet code 2
remained unused. Clip 2.0 improved from 2.254187 to **2.226880**, beating the
mature clip-3 control at 2.251191 and the clip-2.5 arm by 0.000836. Its route
distribution retains 84.95% zeros while assigning 4.74% to code 2, so all four
probability codes are active. This is the first measured attention refinement
that improves bounded loss without replacing one collapsed alphabet with
another.

The predeclared 5,000-step refinement is now complete. It was not monotonically
better: its periodic 200-batch loss moved from **2.255291** at step 2,000 to
**2.240107** at step 4,000, then regressed to **2.249037** at step 5,000.
Exhaustive sequential evaluation of the fixed-budget endpoint nevertheless
measured **2.235370 loss / 9.34994 perplexity** over 4,907,776 targets. That is
0.005600 lower than the previous best strict exhaustive result of 2.240969.
All four route codes remained active at 85.69%, 8.26%, 4.36%, and 1.69%;
attention entropy was 2.1434 and residual normalized MSE was 2.28%.

The handoff guard also did its job. A separate matched re-evaluation scored the
original screen checkpoint at **2.226880** and the refinement's saved best at
**2.240827**, so the screen checkpoint—not the longer run—supplies the next
shared-scale experiment. The fixed 5,000-step endpoint remains a legitimate new
exhaustive best and is preserved separately. Its generations are recognizably
story-like but still contain pronoun swaps, repetition, malformed words, and
unfinished endings; the loss improvement is not language-quality parity. The
run, packed export, hashes, and audit are byte-verified on
[Hugging Face](https://huggingface.co/sanjuhs/ternary-llm-experiment/tree/main/tinystories-28m/runs/attention-clip-selected-2.0-refine),
and the full clip comparison is published
[separately](https://huggingface.co/sanjuhs/ternary-llm-experiment/tree/main/tinystories-28m/comparisons/attention-clip-refinement).

The remaining pipeline now preserves two different endpoints instead of
quietly conflating them. The quality branch may retain token-wise QKV scales,
GELU, or floating RMSNorm when a stricter replacement loses too much. A
separate mandatory strict-contract branch starts from the learned per-head
scale checkpoint and combines ternary Q/K/V, the two-bit integer route, ReLU,
and exact integer-reference RMSNorm. Its exporter must report
`ternary_operand_contract.satisfied = true` or the stage fails. This still does
not claim a fused end-to-end integer runtime: requantization-scale arithmetic
and final token sampling remain explicit outstanding boundaries.

Together, these papers suggest two honest follow-ups. For exact two-bit storage,
improve the two binary planes with block reconstruction, groupwise fixed-point
scales, and layer sensitivity training. For quality with ternary operators,
allow additional ternary planes or expanded ternary factors and measure the
extra storage and additions explicitly.

## Residual refinement: the first matched result

We implemented the R2Q-style idea at activation boundaries and evaluated six
equal-budget arms. Two binary planes are exactly two code bits per activation
scalar. Ternary planes keep every matrix operand in `-1/0/+1`, but ordinarily
occupy two physical bits per plane.

| Projection | Activation planes | Physical bits | Loss after QAT | Residual normalized MSE |
|---|---|---:|---:|---:|
| Learned COAT | 2 binary | 2 | 4.339853 | 12.62% |
| Fixed Hadamard | 2 binary | 2 | **4.332306** | 12.47% |
| Learned COAT | 2 ternary | 4 | 3.295858 | 4.30% |
| Fixed Hadamard | 2 ternary | 4 | **3.288210** | 4.34% |
| Learned COAT | 3 ternary | 6 | 2.664104 | 1.66% |
| Fixed Hadamard | 3 ternary | 6 | **2.661332** | 1.68% |

The fixed Hadamard transform slightly beat learned COAT after training in every
matched row. The differences are small, but their direction is consistent and
valuable: we can retain an add/subtract-only transform rather than paying for a
dense learned rotation.

The dominant variable is representation error. Reducing residual error from
about 12.5% to 4.3% and 1.7% moves loss from 4.33 to 3.29 and 2.66. The
three-plane model improves the old one-code result by 2.44 loss, yet remains
behind the 2.1126 A4 control and costs six physical code bits. It is therefore
the best ternary-operator result, not the best storage result.

The extended exact-two-bit run completed another 5,000 QAT steps and
moved loss from 4.332306 to **4.258285**. The final 2,000 steps recovered only
0.005699, establishing a practical plateau. The two-bit representation is
stable and trainable, but its current per-token, two-plane lattice discards too
much residual information. Groupwise fixed-point scales, block reconstruction,
or a residual-free architecture are now better-founded changes than simply
training the same graph longer.

## What the survivor experiments changed

We ran the longer three-plane model, an exact three-binary-plane control, two
equal-storage layer allocations, and a GELU-to-ReLU hardening stage. Every
survivor was then evaluated over all 4,907,776 validation targets.

| Activation route | Physical bits per residual scalar | Full loss | Perplexity |
|---|---:|---:|---:|
| Two binary planes | **2** | 4.251708 | 70.2252 |
| Three binary planes | 3 | 3.750950 | 42.5615 |
| Three ternary planes, extended | 6 | 2.598547 | 13.4442 |
| Late-layer ternary mix `[2,2,2,3,3,3]` | 5 average | 2.889333 | 17.9813 |
| NMSE-aware ternary mix `[3,2,3,2,2,3]` | 5 average | 2.858985 | 17.4438 |
| Three ternary planes with ReLU | 6 | **2.518279** | **12.4072** |

Three conclusions survive the exhaustive evaluation.

First, exact two-bit storage is a genuine operating point, but not yet a
quality-preserving one. Moving to three binary code bits recovers 0.50076 loss,
yet remains far from the ternary-plane route.

Second, layer allocation should follow measured quantization error. With the
same average five physical bits, putting third planes in layers 0, 2, and 5
beats putting them only in the final three layers by 0.03035 loss. That is the
small-model version of the sensitivity-aware allocation advocated by TWLA and
RobuQ.

Third, ReLU is unexpectedly beneficial. Replacing GELU and adapting for 1,500
steps improves the extended three-plane model by 0.08027 loss. ReLU is also
friendlier to integer or comparator-based hardware. This removes one
floating-point-looking operation while improving the language-model objective.

The result is still not float parity: it spends six physical activation bits
and remains above the A4 control. It is, however, the strongest evidence so far
that a ternary-operand Transformer should be co-designed around its discrete
forward path rather than obtained by mechanically rounding a conventional
GELU Transformer.

## The matched float model clears the reference

The 27.4M-parameter float control completed a token budget equal to two passes
over the 488M-token corpus. Training windows were sampled randomly, so
“two-pass” is a budget label rather than a deterministic sequential epoch.

The exhaustive result is **1.344198 loss** and **3.8351 perplexity** over
4,907,776 validation targets. On the same raw validation text this corresponds
to approximately **0.495229 bits per UTF-8 byte**, compared with **0.553544**
for the released TinyStories-33M checkpoint. The new control is about 10.5%
better on that tokenizer-neutral metric.

This changes the interpretation of the project. The old 5.84M teacher was a
capacity confound; the new one is not. Any remaining ternary loss gap can now be
attributed to conversion, discrete optimization, or architecture with much
greater confidence. The automatically queued stages start with one
corpus-equivalent ternary-weight QAT budget, followed by packed export, A4
activation adaptation, and the strict ternary-QKV/low-bit-attention/ternary-
residual stage.

## The matched ternary-weight model

After one corpus-equivalent QAT budget, the 27.4M model with ternary forward
weights reaches **1.535463 loss** and **4.6435 perplexity** on the exhaustive
validation stream. Its approximately **0.565695 bits per UTF-8 byte** is only
2.2% worse than the released TinyStories-33M checkpoint, but 14.2% worse than
its own float teacher's BPB. Those are both useful comparisons: it is already a
strong tiny language model, while the matched teacher still exposes a real
conversion gap.

The weight-code distribution is 32.61% negative, 35.05% zero, and 32.34%
positive. The inference export occupies **7,509,079 bytes**, compared with
337,312,859 bytes for the resumable checkpoint containing float shadow weights
and optimizer state. That 44.9x file-size difference is not a pure tensor
compression benchmark, but it verifies that the deployed weight payload is
compact and the training payload is correctly kept out of inference.

The next stage is the matched COAT A4 activation control. It keeps the same
ternary weights and teacher so we can measure the activation cost separately
before forcing ternary Q/K/V, the integer attention route, and ternary residual
planes.

## The matched A4 control does not preserve loss

After 15,000 adaptation steps, calibrated COAT A4 activations reach
**1.800118 loss** and **6.0504 perplexity** on the exhaustive validation stream.
That is 0.264656 worse than the matched ternary-weight checkpoint.

This matters because the small-model A4 experiments looked much closer to their
local controls. At 27.4M parameters and a stronger teacher, the activation
boundary is plainly the dominant remaining error source. COAT's learned
orthogonal projection helps organize outliers, but it cannot by itself recover
information discarded by a four-bit scalar codebook.

The final running stage is therefore a stress test, not an assumed win. It
starts from this A4 checkpoint and forces ternary Q/K/V, the four-entry integer
attention lookup, two-bit attention routes, fixed Hadamard mixing, and three
ternary residual planes. Its purpose is to measure the real cost of the
hardware-oriented graph at matched capacity.

## Shared QKV scales expose the hardware-quality price

The first large-model ASIC-hardening comparison is complete. Both arms started
from the same clip-2 checkpoint and received 4,000 QAT steps. The ordinary arm
computed a fresh scale for each token; the hardware arm learned one stable
scale for every Q/K/V head, allowing scale multiplication to move outside the
large ternary dot products.

On the exhaustive 4,907,776-target validation stream, dynamic token scales
reach **2.230903 loss / 9.3083 perplexity**. Learned head scales reach
**2.309650 / 10.0709**. The 0.078748 gap is the measured price of the cleaner
factorization in this experiment. It is not attention collapse: the learned
arm still uses all four routing codes, with 85.19% zero, 4.39% code 2, and
1.96% code 3, and its attention entropy is 2.4293.

The dynamic control is now our best exhaustive code-constrained result, beating
the longer clip-2 refinement's 2.235370 by 0.004467. It still fails the strict
deployment contract because its data-dependent scales are not ternary operands.
The learned arm is more hardware-friendly but not selected for the quality
branch. A fresh identical 200-batch gate retained the untouched source at
2.226880 rather than either trained arm at 2.241261 and 2.321006. This is why
we keep separate quality and mandatory-hardware endpoints.

Qualitative samples agree with the loss ordering. The token-scale model writes
recognizable multi-sentence stories but still repeats objects and occasionally
breaks dialogue. The learned-scale model is also readable, yet shows more
repetition and semantic errors. These samples are useful diagnostics, not a
replacement for held-out loss. Both audited runs, packed exports, generations,
and the comparison metadata are preserved in the public Hugging Face model
repository.

## An explicit no-update route is valid, but does not win

The next matched experiment replaced ordinary integer-LUT Softmax with
Softmax-1: each query receives a virtual route that contributes to the integer
denominator but adds no value vector. This gives attention a legitimate “send
nothing” choice without a floating bypass.

Before adaptation, Softmax-1 scored 2.268965 versus 2.226880 for ordinary
Softmax. After 4,000 equal-budget QAT steps, the matched losses were 2.252379
and 2.241261. Exhaustive validation confirmed the ordering:

| Normalization | Exhaustive loss | Perplexity | No-update mass |
|---|---:|---:|---:|
| Ordinary integer Softmax | **2.230903** | **9.3083** | approximately 0% |
| Integer Softmax-1 | 2.241651 | 9.4089 | **1.54%** |

Softmax-1 learned to use the new route and recovered most of its initial gap,
but remained 0.010748 loss worse. This rejects the hypothesis for the current
quality branch without invalidating the integer mechanism. Ordinary Softmax is
selected; the Softmax-1 run, generations, packed export, and comparison are
checksum-verified and published on Hugging Face.

## ReLU is simpler and unexpectedly improves the large model

GELU is a smooth curve. It works well in ordinary Transformers, but a ternary
accelerator would need a lookup table or a higher-precision approximation to
compute it. ReLU is only a sign test: negative values become zero and positive
values pass through. That makes it a much cleaner target for a low-bit
inference graph.

We did not assume that simpler meant better. GELU and ReLU started from the
same untouched checkpoint, saw the same examples in the same order, used the
same teacher losses, and received exactly 4,000 adaptation steps. Their
exhaustive results over 4,907,776 held-out targets were:

| Feed-forward function | Loss | Perplexity |
|---|---:|---:|
| GELU control | 2.230903 | 9.3083 |
| ReLU | **2.160893** | **8.6789** |

ReLU improves loss by **0.070009**, so it wins on both simplicity and measured
quality. The attention system remains active rather than collapsing: its
entropy is 2.1415, all four route codes occur, and the largest zero-route share
is 85.57%. Residual reconstruction normalized MSE is 0.02242.

This is now the best exhaustive code-constrained model in the project, but the
word “constrained” needs care. Its large matrix weights, Q/K/V codes, attention
routes, and three residual planes are low-bit, while its per-token QKV scales
are still data-dependent. The export therefore correctly reports that the
strict ternary-operand contract is not satisfied. The separate hardware branch
below combines ReLU with learned factorizable head scales and integer-reference
RMSNorm, and reports its loss independently rather than hiding it behind the
better quality-branch number.

### Integer RMSNorm preserves the quality result

RMSNorm looks simple, but its square, mean, reciprocal square root, and
rescaling are another place where a nominally low-bit Transformer can quietly
fall back to floating point. We evaluated an integer-reference implementation
against the same saved ReLU checkpoint without any additional training:

| RMSNorm arithmetic | 200-batch loss | Exhaustive loss | Perplexity |
|---|---:|---:|---:|
| Float reference | 2.171209 | 2.160893 | 8.6789 |
| Integer reference | **2.170885** | **2.160526** | **8.6757** |

The exhaustive change is **-0.000367**, effectively neutral and slightly
favourable on this validation stream. This removes RMSNorm from the quality
endpoint's list of floating-point exceptions. Its remaining failed
ternary-operand-contract check is the data-dependent QKV scale; the strict
hardware branch below tests a factorizable per-head scale separately.

### The strict ternary-operand endpoint passes

The separate hardware branch starts from the learned per-head-scale checkpoint
and adapts it for 4,000 steps with ternary Q/K/V, all four integer attention
codes, three ternary residual planes, fixed Hadamard mixing, ReLU, and the same
teacher objectives. It improves monotonically on the matched validation gates:

| Evaluation | Loss | Perplexity |
|---|---:|---:|
| Step 2,000, 200 batches | 2.278716 | 9.7641 |
| Step 4,000, 200 batches | 2.262977 | 9.6117 |
| Exhaustive, float RMSNorm | 2.250051 | 9.4882 |
| Exhaustive, integer RMSNorm | **2.250638** | **9.4938** |

Integer RMSNorm adds only **0.000586** loss on this stricter activation
distribution. The packed export passes all eleven ternary-operand checks and
contains 59 scaled-ternary tensors, eight fixed-point scale tensors, and eight
readiness buffers. Attention does not collapse: at step 2,000 its four route
codes occupy 86.39%, 7.61%, 4.13%, and 1.87% of valid positions, and the
ternary weights are almost evenly divided among negative, zero, and positive
codes.

This result is a real endpoint, but it is not float parity. It scores 0.090112
worse than the dynamic-token-scale quality model and 0.906440 worse than the
ordinary float teacher baseline. Three ternary residual planes
also require six physical code bits per scalar. Finally, the PyTorch reference
still uses emulated fixed-point scale/requantization arithmetic and a final
sampling Softmax. The correct claim is therefore **strict ternary matrix
operands and persistent boundaries**, not “every temporary scalar is
ternary.”

## Binary Q/K loses information that ternary Q/K needs

BWTA and BinaryAttention make a useful hardware argument: binary
`{-1,+1}` query and key vectors are especially cheap, and the value vectors can
remain ternary. Binary arithmetic is also a subset of a ternary accelerator.
The unresolved question was whether removing Q/K's zero code would make
attention denser and recover quality.

We tested that hypothesis against a matched ternary-Q/K/V control. Both arms
started from the ReLU checkpoint, received 4,000 steps, and used the same
teacher, examples, optimizer, and Q/K-relation distillation loss:

| Q/K/V alphabet | Before adaptation | Exhaustive loss | Perplexity |
|---|---:|---:|---:|
| Ternary Q/K/V | **2.171209** | **2.171827** | **8.7743** |
| Binary Q/K, ternary V | 2.333700 | 2.307770 | 10.0520 |

The binary arm learns—it improves by 0.025929 between its initial 200-batch
screen and exhaustive endpoint—but it remains **0.135944** worse than the
matched ternary control. It also remains 0.146877 worse than the protected
ReLU source's exhaustive loss. The result says that zero is not merely wasted
Q/K capacity in this network; it carries useful attention geometry.

There is a second, quieter result. Adding the Q/K-relation objective to the
ordinary ternary control makes its exhaustive loss 0.010933 worse than the
source. The source-selection guard therefore rejects both newly trained arms
and retains the original ReLU checkpoint. This prevents a well-motivated
distillation term from silently degrading the quality branch. The mandatory
hardware endpoint also continues to use full ternary Q, K, and V, exactly as
required by the project contract.
