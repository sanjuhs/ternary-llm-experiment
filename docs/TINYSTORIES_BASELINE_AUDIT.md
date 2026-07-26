# TinyStories baseline audit

*Protocol frozen July 26, 2026*

## The short answer

There is no single canonical “TinyStories loss.” Loss depends on the tokenizer,
validation split, story-boundary tokens, context length, and whether the model
was trained for one epoch or six. The original
[TinyStories paper](https://arxiv.org/abs/2305.07759) released small models and
evaluated their generated stories, but did not publish one standard validation
loss that every reproduction can target.

Three numbers are commonly confused:

| Source | Reported loss | Why it is not directly interchangeable |
|---|---:|---|
| Community 28M reproduction | about **1.32** | Two epochs and an incompletely specified evaluation protocol |
| “GPT-2 30M” model card | **1.272** | Cleaned TinyStoriesV2, six epochs, GPT-2 tokenizer; the card also reports 49.5M parameters including embeddings and the Hub reports 58.9M tensors |
| Official released TinyStories-33M, measured here | **1.54010** | Original validation text, released GPT-2 tokenizer, context 256, 99.996% token coverage |

The first figure comes from the
[gpt-tinystories reproduction](https://github.com/raymond-van/gpt-tinystories).
The second comes from the
[GPT-2 30M TinyStories model card](https://huggingface.co/0rn0/gpt2-30m-tinystories).
They are useful landmarks, not matched baselines.

## Our reproducible comparison

We downloaded the released `roneneldan/TinyStories-33M` checkpoint and evaluated
it on all 21,990 stories in the official validation split. We also scanned our
current floating-point checkpoint over the complete token stream instead of
sampling 100 random batches.

| Model | Vocabulary | Parameters | Training budget | Loss | Perplexity | Bits / UTF-8 byte |
|---|---:|---:|---:|---:|---:|---:|
| Released TinyStories-33M | 50,257 | about 68.5M learned tensors including embeddings | Released checkpoint | **1.54010** | **4.6650** | **0.55354** |
| Our float baseline | 4,096 | 5,836,032 | 488,177,664 sampled tokens, about one corpus pass | **1.68961** | **5.4174** | **0.62249** |

Raw token loss and perplexity are not comparable across different tokenizers.
Bits per UTF-8 byte is the fairer common-text metric. On that measure, our
existing float model is about **12.5% behind** the released checkpoint.

For our 4,096-token tokenizer, **1.50247 loss** corresponds to the released
model's 0.55354 bits-per-byte rate. That is the capacity-control target for the
new 27.4M-parameter run. It is a derived parity threshold, not a reported paper
number.

The evaluator is reproducible:

```bash
uv sync --extra reference
uv run python scripts/evaluate_tinystories_reference.py \
  --model artifacts/reference/TinyStories-33M \
  --validation-parquet data/raw/validation/*.parquet \
  --context-length 256 \
  --batch-size 16 \
  --device mps

uv run ternary-evaluate \
  --checkpoint artifacts/full-stage-a/float/checkpoint.pt \
  --config configs/full.toml \
  --device mps \
  --sequential
```

## Why the existing baseline does not match

The current baseline is a 5.84M-parameter model trained for one nominal pass.
The public models behind the 1.32 and 1.272 headlines use larger bodies, much
larger embedding tables, more epochs, a different tokenizer, or a cleaned
dataset. A quantized student cannot be expected to reach their score until its
own floating-point capacity control reaches the matched target.

The new control has 27,402,752 learned parameters, width 512, eight layers,
eight heads, context 256, and the same 4,096-token tokenizer as our experiment.
It is scheduled for two full corpus passes. That isolates model capacity and
training budget before introducing ternary weights.

## The inference rule

Training may retain floating shadow weights, gradients, optimizer moments, and
a floating teacher. The deployed forward graph has the stricter contract:

1. all learned matrix weights are ternary codes;
2. Q, K, and V are ternary;
3. attention scores use four codes and the exponential uses a four-entry
   integer lookup table;
4. normalized attention routing is low-bit;
5. every persistent residual boundary is encoded as binary or ternary planes;
6. matrix products consume binary/ternary codes and use wider integer
   accumulators;
7. scales and reductions are fixed-point candidates and are requantized before
   the next persistent boundary;
8. there is no floating residual or low-rank bypass at inference.

The present PyTorch implementation is a fake-quantization reference for quality
experiments. It proves the code alphabet and forward values, not yet a packed
end-to-end ASIC kernel. A packed reference export and an integer-equivalence
test remain separate deployment milestones.

## Why residual planes are the next experiment

One ternary code per hidden feature scored 5.0989 loss after the progressive
curriculum. The sharp quality break occurred below seven representable levels.
An orthogonal transform can redistribute information, but it cannot restore
information discarded by a three-level scalar bottleneck.

[R2Q](https://arxiv.org/abs/2511.21736) represents a two-bit weight as two
successive binary refinements:

```text
value ≈ scale₁ × binary_code₁ + scale₂ × binary_code₂
```

We adapt that representation to activations. This adaptation is a hypothesis;
R2Q itself evaluates weights. Two binary planes require exactly two code bits
per scalar, and binary arithmetic is a subset of ternary arithmetic. Three
sparse ternary planes cost more bits but test whether explicit zero codes and
additional residual capacity recover quality.

The current experiment matrix is:

| Projection | Activation representation | Logical code budget | Purpose |
|---|---|---:|---|
| Learned COAT basis | 2 binary planes | 2.00 bits | Best-quality rotation with exact two-bit codes |
| Fixed Hadamard | 2 binary planes | 2.00 bits | Strict ASIC-friendly two-bit route |
| Learned COAT basis | 2 sparse ternary planes | 3.17 bits | Test the value of explicit zero codes |
| Fixed Hadamard | 2 sparse ternary planes | 3.17 bits | Strict projection counterpart |
| Learned COAT basis | 3 sparse ternary planes | 6 physical bits | Capacity upper control |
| Fixed Hadamard | 3 sparse ternary planes | 6 physical bits | Strict quality-oriented route |

Every arm keeps ternary weights, ternary Q/K/V, and the integer-LUT attention
path. The fixed Hadamard route uses only signs, additions, and a known
normalization scale. A learned dense COAT rotation is a quality control, not the
final strict operator.

## What the newest papers change

- [BitNet v2](https://arxiv.org/abs/2504.18415) is strong evidence for online
  Hadamard mixing. Its W1.58A4 ablation without rotation diverged, while the
  rotated model remained stable. It does not establish A2 or ternary residuals.
- [TWLA](https://arxiv.org/abs/2606.13054) combines asymmetric ternary weight
  relocation, Kronecker orthogonal shaping, and adjacent-layer-aware activation
  bit allocation. It reaches W1.58A4, not uniformly ternary activations.
- [RobuQ](https://arxiv.org/abs/2509.23582) demonstrates stable average-A2
  diffusion Transformers. Its mixed precision and low-rank floating branch do
  not satisfy our inference rule, but its sensitivity result is valuable:
  attention output projections and later blocks deserve more representation
  capacity.
- [The Quantization Benefits of Residual-Free
  Transformers](https://arxiv.org/abs/2605.25880) supports our observed failure
  mode: residual mixing produces heavy-tailed activations and amplifies
  low-bit error. A residual-free TinyStories architecture is the next
  architecture-level arm if residual planes cannot close the gap.
- [BWTA](https://arxiv.org/abs/2604.03957) supplies the closest
  binary-weight/ternary-activation training and kernel blueprint, including a
  staged alphabet reduction and magnitude alignment. Its published LLM route
  is not uniformly low-bit across all layers.

## Decision gates

The experiment will be reported as successful only if:

- the 27.4M float control reaches the tokenizer-neutral reference range;
- a ternary-weight control is evaluated with the same checkpoint, split, and
  token budget;
- the strict residual-plane model substantially improves on 5.0989;
- fixed prompts generate coherent, repeatable TinyStories-like text;
- diagnostics confirm the advertised code alphabets;
- the final packed/integer reference matches fake-quantized logits within a
  declared tolerance.

Until those gates pass, “fully ternary” means an executed research hypothesis,
not a solved replacement for floating-point GPT.

The small-model survivor audit has now completed. Its best exhaustive result is
the ReLU-hardened three-ternary-plane model at loss **2.518279** and perplexity
**12.4072**. The exact two-bit binary-plane model plateaus at loss **4.251708**.
Those results validate the discrete training machinery, but neither answers the
capacity gate. The 27.4M float control is running over the full 488M-token
corpus before its matched ternary conversions.
