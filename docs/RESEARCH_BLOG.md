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
