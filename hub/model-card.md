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

These are small research models, not production assistants. See the GitHub research
blog for limitations and activation-quantization results.
