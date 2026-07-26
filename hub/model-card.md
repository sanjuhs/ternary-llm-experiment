---
license: mit
library_name: pytorch
tags:
  - ternary
  - quantization
  - tinystories
  - research
datasets:
  - roneneldan/TinyStories
---

# Ternary LLM Experiment artifacts

This repository stores checkpoints, compact packed exports, projection
calibrations, metrics, and samples for
[`sanjuhs/ternary-llm-experiment`](https://github.com/sanjuhs/ternary-llm-experiment).

The main completed models are 5.84M-parameter GPT-style TinyStories models trained
for one nominal pass over 488M sampled tokens:

| Mode | Validation loss | Perplexity |
|---|---:|---:|
| Float, 100-batch historical evaluation | 1.6989 | 5.47 |
| Float, exhaustive sequential evaluation | **1.68961** | **5.4174** |
| Ternary weights / float activations | 2.0312 | 7.62 |

The `model-2bit.pt` file is an inference-oriented export containing packed ternary
codes and scales. The larger `checkpoint.pt` files are resumable research
checkpoints that include shadow weights and optimizer state.

## Two-bit attention pilot

Each attention arm starts from the same COAT-A4 checkpoint and receives 500
quantization-aware fine-tuning steps. Evaluation uses 100 deterministic batches
from the full 4.9M-token validation stream.

| Attention representation | Validation loss | Perplexity |
|---|---:|---:|
| Float attention control | 2.1132 | 8.275 |
| 2-bit softmax input | 2.1489 | 8.576 |
| 2-bit normalized probability | 2.1383 | 8.485 |
| 2-bit input + probability | 2.1569 | 8.644 |
| Binary probability, best threshold | 2.1889 | 8.926 |

The `attention-pilot/` directory contains all six resumable checkpoints,
resolved configurations, metrics, and the complete pre-fine-tuning sweep.
“Two-bit” refers to stored operands and boundary codes; reductions use wider
accumulators, as required to avoid overflow.

These are small research models, not production assistants. See the GitHub research
blog for limitations and activation-quantization results.

## Matched TinyStories baseline audit

The original TinyStories paper does not publish one canonical validation loss.
The public “28M” reproduction reports about 1.32 after two epochs, while a
separate “GPT-2 30M” model card reports 1.272 after six epochs on a cleaned
dataset and actually lists 49.5M parameters including embeddings.

We evaluated the released `roneneldan/TinyStories-33M` checkpoint and our float
checkpoint exhaustively on the official validation text. Because their
tokenizers differ, bits per UTF-8 byte is the meaningful common metric:

| Model | Token loss | Perplexity | Bits / UTF-8 byte |
|---|---:|---:|---:|
| Released TinyStories-33M | 1.54010 | 4.6650 | **0.55354** |
| Our 5.84M float baseline | 1.68961 | 5.4174 | **0.62249** |
| Our 27.4M matched float control | **1.34420** | **3.8351** | **0.49523** |
| Our 27.4M ternary-weight model | **1.53546** | **4.6435** | **0.56570** |

The exact JSON records are in `baseline-audit/`. The full methodology and
comparability caveats are in the GitHub
[`TINYSTORIES_BASELINE_AUDIT.md`](https://github.com/sanjuhs/ternary-llm-experiment/blob/agent/gated-ternary-attention/docs/TINYSTORIES_BASELINE_AUDIT.md).

The 27.4M control trained for 976,355,328 randomly sampled tokens, a
two-corpus-equivalent token budget. It beats the released 33M checkpoint by
about 10.5% on the tokenizer-neutral bits-per-byte metric. Its checkpoint,
exhaustive evaluation, diagnostics, generations, and metrics are in
`tinystories-28m/float-two-pass/`.

The matched ternary-weight model uses 35.05% zero codes and is only 2.2% behind
the released 33M checkpoint in BPB, while remaining 14.2% behind its own float
teacher. Its `model-2bit.pt` inference export is 7,509,079 bytes; the full
resumable shadow-weight checkpoint is intentionally much larger.

## Residual-plane refinement

Every arm below retains ternary weights, ternary Q/K/V, and the low-bit integer
attention route. “Physical bits” counts ordinary two-bit packing per ternary
plane; it does not substitute entropy for actual storage.

| Activation representation | Physical code bits | Exhaustive full loss |
|---|---:|---:|
| Two binary planes, fixed Hadamard | **2** | 4.25171 |
| Three binary planes, fixed Hadamard | 3 | 3.75095 |
| Three ternary planes, extended | 6 | 2.59855 |
| NMSE-aware ternary layer mix | 5 average | 2.85899 |
| Three ternary planes, ReLU-hardened | 6 | **2.51828** |

The exact-two-bit run plateaued. Error-aware layer allocation beats a
late-layer allocation at equal average storage, and replacing GELU with ReLU
improves the best ternary-plane model. These checkpoints, exhaustive
evaluations, diagnostics, generations, metrics, and resolved configurations are
in `residual-refinement-pilot/`. Multi-plane ternary models preserve ternary
matrix operands but are not 1.58-bit-per-activation models.
