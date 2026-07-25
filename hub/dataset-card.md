---
license: cdla-sharing-1.0
task_categories:
  - text-generation
language:
  - en
tags:
  - tinystories
  - tokenized
  - ternary
---

# TinyStories 4K-BPE token stream

This is a deterministic, tokenized derivative used by the Ternary LLM Experiment.
The source is
[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories).

- tokenizer: byte-level BPE, vocabulary 4,096
- token dtype: little-endian `uint16`
- training stories: 2,119,719
- training tokens: 488,174,163
- validation stories: 21,990
- validation tokens: 4,907,807

Each story is encoded with explicit beginning/end markers and concatenated into
`train.bin` or `validation.bin`. `metadata.json` records the preprocessing
parameters. The source repository contains the reader and exact preparation code.

The original dataset's terms and attribution continue to apply. This repository is
provided for experiment reproducibility, not as a replacement for the source
dataset card.
