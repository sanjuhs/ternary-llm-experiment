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

[BitNet v2](https://arxiv.org/abs/2504.18415) shows that online Hadamard mixing
can stabilize W1.58A4 training where the corresponding unrotated A4 treatment
diverges. It supports our fixed-Hadamard arm, but it does not establish A2.

[R2Q](https://arxiv.org/abs/2511.21736) decomposes a two-bit weight into two
successive binary residual kernels. Our exact-two-bit activation arm adapts that
representation to residual states. That activation use is our hypothesis, not a
result claimed by R2Q.

[TWLA](https://arxiv.org/abs/2606.13054) combines asymmetric ternary relocation,
Kronecker orthogonal shaping, and adjacent-layer-aware mixed precision. Its
strongest lesson for this project is that later or sensitive blocks should
receive more activation capacity. It still targets W1.58A4 rather than a
uniformly ternary residual stream.

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
1.5, 2.0, and 2.5, adapts each for the same token budget, selects on the same
held-out batches, and exhaustively evaluates a longer refinement of the winner.
It is a targeted test of codebook utilization, not an after-the-fact change to
the already-running clip-3 baseline.

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
