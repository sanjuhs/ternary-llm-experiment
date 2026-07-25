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
| Float | 1.6989 | 5.47 |
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
