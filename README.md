# Ternary LLM Experiment

For the complete result inventory and the next ternary-QKV architecture, see
[the experiment ledger and roadmap](docs/EXPERIMENT_LEDGER_AND_ROADMAP.md).
For the published 28M/30M TinyStories claims and our tokenizer-neutral matched
evaluation, see the
[TinyStories baseline audit](docs/TINYSTORIES_BASELINE_AUDIT.md).

This repository tests whether a small GPT-style language model can learn TinyStories
while its learned weights and persistent forward activations use ternary codes
`{-1, 0, +1}`.

The exact hypothesis, precision boundary, ablations, metrics, and later discrete
learning stages are defined in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). The raw
conversation that motivated the project is preserved in [idea.md](idea.md).
The results and COAT follow-up are explained for a broader audience in
[docs/RESEARCH_BLOG.md](docs/RESEARCH_BLOG.md). For a gentler introduction, read
[docs/SIMPLIFIED_TRANSFORMER_BLOG.md](docs/SIMPLIFIED_TRANSFORMER_BLOG.md).
The exact meaning of “completely quantized” is defined in
[docs/COMPLETE_QUANTIZATION_CONTRACT.md](docs/COMPLETE_QUANTIZATION_CONTRACT.md).
Matched fixed-prompt outputs are preserved in
[docs/ATTENTION_GENERATION_SAMPLES.md](docs/ATTENTION_GENERATION_SAMPLES.md);
the newer ternary-QKV and integer-LUT outputs are in
[docs/GATED_ATTENTION_GENERATION_SAMPLES.md](docs/GATED_ATTENTION_GENERATION_SAMPLES.md).
The strict residual-curriculum outputs are preserved in
[docs/FULLY_TERNARY_GENERATION_SAMPLES.md](docs/FULLY_TERNARY_GENERATION_SAMPLES.md).

## What is implemented

- reproducible `uv` environment;
- TinyStories download, deterministic selection, BPE tokenizer training, and binary
  token shards;
- a configurable decoder-only transformer;
- `float`, `ternary_weights`, `ternary_activations`, `ternary_forward`, and
  population-coded ternary modes;
- Hadamard and calibrated COAT-style residual projections with matched A4 and
  ternary-activation arms;
- four-code attention-score, four-code attention-probability, combined, and
  binary-routing experiments with code-use diagnostics;
- independently forced ternary Q/K/V, Q-ViT-style rectification, binary no-op
  gates, and an integer four-entry exponential lookup reference;
- a BWTA-inspired progressive residual alphabet with magnitude alignment,
  per-layer code-use diagnostics, hidden-state distillation, and an exact
  three-code endpoint;
- exact two-bit binary residual refinement, multi-plane ternary residuals,
  fixed-Hadamard mixing, and layer-specific plane allocation;
- teacher/student logit, attention-map, and sampled Q-Q/K-K relation
  distillation;
- exact reference linear, attention Q·K, and two-plane Route·V paths with
  binary/ternary code operands and INT32 accumulators;
- an integer-LUT Softmax-1 route with a virtual no-update code that contributes
  to the integer denominator without adding a value vector;
- straight-through ternary fake quantization with per-row weight and per-token
  activation scales;
- training, validation, checkpoint/resume, and text generation;
- unit tests and laptop-sized smoke/research configurations.

The current code emulates ternary arithmetic with PyTorch tensors. It measures the
learning behavior of ternary representations. A portable two-bit packed-weight
reference validates storage and numerical correctness, but it deliberately does
not claim a speedup without a fused device kernel.

## Published artifacts

- [Model checkpoints, packed weights, projections, and
  metrics](https://huggingface.co/sanjuhs/ternary-llm-experiment)
- [Ternary-QKV and strict integer-LUT pilot
  artifacts](https://huggingface.co/sanjuhs/ternary-llm-experiment/tree/main/gated-attention-pilot)
- [Progressive fully ternary residual pilot
  artifacts](https://huggingface.co/sanjuhs/ternary-llm-experiment/tree/main/fully-ternary-pilot)
- [Residual-plane refinement
  artifacts](https://huggingface.co/sanjuhs/ternary-llm-experiment/tree/main/residual-refinement-pilot)
- [Complete tokenized TinyStories
  stream](https://huggingface.co/datasets/sanjuhs/ternary-tinystories-4096)

Generated data and checkpoints remain ignored by Git; the public Hub repositories
hold the experiment checkpoints and reproducible token streams.

Completed run directories can be audited, uploaded, and remotely enumerated in
one command. The command writes a deterministic `artifact-manifest.json` before
uploading and fails if any local file is absent from the resulting Hub commit:

```bash
uv run ternary-publish-hf \
  artifacts/tinystories-28m/ternary-weights-one-pass \
  --path-in-repo tinystories-28m/ternary-weights-one-pass \
  --ipv4-only
```

## Setup

`uv` manages the Python interpreter and virtual environment:

```bash
uv sync --extra dev
uv run pytest
```

## Data

The Hugging Face dataset contains about 2.14 million stories and about 1 GB of
Parquet data. Download it into the repository-local cache:

```bash
uv run ternary-data download
```

Prepare the bounded first experiment (50,000 train and 2,000 validation stories):

```bash
uv run ternary-data prepare \
  --max-train-stories 50000 \
  --max-validation-stories 2000
```

Prepare all 2.12 million training stories with the same tokenizer:

```bash
uv run ternary-data prepare \
  --full \
  --output-dir data/full \
  --tokenizer-from data/processed/tokenizer.json
```

For the very fast end-to-end smoke run:

```bash
uv run ternary-data prepare \
  --output-dir data/smoke \
  --max-train-stories 200 \
  --max-validation-stories 50 \
  --vocab-size 512
```

`download` writes size-checked Parquet shards under `data/raw`. `prepare` uses
`data/huggingface` for its local Arrow cache, then writes a tokenizer, `uint16`
token streams, and metadata under the selected output directory.

## Train

Smoke:

```bash
uv run ternary-train --config configs/smoke.toml
```

Research configuration:

```bash
uv run ternary-train --config configs/tiny.toml --mode float --run-name baseline
uv run ternary-train --config configs/tiny.toml --mode ternary_weights --run-name weights
uv run ternary-train --config configs/tiny.toml --mode ternary_activations --run-name activations
uv run ternary-train --config configs/tiny.toml --mode ternary_forward --run-name strict
```

Full-corpus GPU configuration:

```bash
uv run ternary-train --config configs/full.toml --mode float --run-name float
```

Calibrate a COAT projection from an existing checkpoint and compare A4 with
ternary activations:

```bash
uv run ternary-calibrate-projection \
  --checkpoint artifacts/full-stage-a/ternary_weights/checkpoint.pt \
  --config configs/full.toml \
  --output artifacts/coat/ternary-weights-projection.pt

uv run ternary-evaluate \
  --checkpoint artifacts/full-stage-a/ternary_weights/checkpoint.pt \
  --config configs/full.toml \
  --mode coat_a4 \
  --projection artifacts/coat/ternary-weights-projection.pt

uv run ternary-evaluate \
  --checkpoint artifacts/full-stage-a/ternary_weights/checkpoint.pt \
  --config configs/full.toml \
  --mode coat_ternary \
  --projection artifacts/coat/ternary-weights-projection.pt
```

`coat_*` modes refuse to run without a calibrated projection. `hadamard_*` modes
use a fixed normalized Hadamard matrix as the data-independent control.

Evaluate two-bit attention boundaries while keeping the same checkpoint and
validation samples:

```bash
uv run ternary-evaluate \
  --checkpoint artifacts/coat-pilot/coat_a4/checkpoint.pt \
  --config configs/full.toml \
  --mode coat_a4 \
  --projection artifacts/coat/ternary-weights-projection.pt \
  --attention-quantization score_int2 \
  --attention-clip 8

# Choices: float, score_int2, prob_int2, score_prob_int2, prob_binary
```

Population-coded residual pilot:

```bash
uv run ternary-train \
  --config configs/full.toml \
  --mode population_ternary \
  --population-lanes 4 \
  --max-steps 1000 \
  --run-name p4
```

Run all four Stage A modes sequentially with:

```bash
scripts/run_stage_a.sh
```

Discrete-learning smoke experiments:

```bash
uv run ternary-train --config configs/counter_smoke.toml
uv run ternary-train --config configs/stochastic_smoke.toml
```

The counter optimizer stores an `int8` evidence counter per parameter; the
stochastic optimizer stores no per-parameter update state. Both constrain the
underlying learnable tensors to scaled ternary values after every update.

## RunPod

The remote workflow uses a checkpointed correctness and cost gate:

```bash
# On the pod after copying the repository to /workspace:
scripts/remote_bootstrap.sh
scripts/remote_benchmark.sh

# Start the complete Stage A matrix only after reviewing benchmark throughput:
scripts/remote_run_stage_a.sh

# Run the matched 2-bit attention PTQ and 500-step QAT matrix:
scripts/remote_attention_pilot.sh

# Run the ternary-QKV, gating, distillation, and integer-LUT pilot:
scripts/remote_gated_attention_pilot.sh

# Progressively reduce every residual boundary to exactly three codes:
scripts/remote_fully_ternary_pilot.sh
```

The full runner resumes any existing per-mode checkpoint. Copy `artifacts/` back to
the local repository before stopping or deleting a pod.

The helper scripts accept the pod host, SSH port, and private-key path:

```bash
scripts/sync_to_runpod.sh HOST PORT PRIVATE_KEY
scripts/sync_from_runpod.sh HOST PORT PRIVATE_KEY
```

## Evaluate and generate

```bash
uv run ternary-evaluate \
  --checkpoint artifacts/smoke/checkpoint.pt \
  --config configs/smoke.toml

uv run ternary-generate \
  --checkpoint artifacts/smoke/checkpoint.pt \
  --tokenizer data/smoke/tokenizer.json \
  --prompt "Once upon a time" \
  --max-new-tokens 80
```

Use `--device cpu`, `--device mps`, or `--device cuda` to override automatic device
selection. Checkpoints include the resolved model and training configuration.

## Packed inference reference

```bash
uv run ternary-packed-benchmark \
  --device cpu \
  --batch 256 \
  --in-features 1024 \
  --out-features 1024
```

The benchmark reports packed storage (including scales) and compares the portable
unpack-then-matmul reference with a dense ternary-weight tensor. It is a correctness
baseline for a future fused kernel, not the fused kernel itself.

On a CUDA PyTorch installation that includes Triton, benchmark the device-native
kernel that decodes packed weights inside the reduction:

```bash
uv run ternary-triton-benchmark \
  --rows 16384 \
  --in-features 256 \
  --out-features 1024
```

Export only the forward ternary codes and scales from a training checkpoint:

```bash
uv run ternary-export \
  --checkpoint artifacts/full-stage-a/ternary_weights/checkpoint.pt \
  --output artifacts/full-stage-a/ternary_weights/model-2bit.pt
```

The `ternary-deployment-v2` artifact packs ternary operands at two bits, stores
learned positive Q/K/V head scales as INT16 fixed-point values, preserves
non-floating buffers, and includes a machine-readable inference-contract
checklist. A packed checkpoint is therefore not automatically labeled
end-to-end integer: the integer RMSNorm reference is not yet wired into the
model runtime, and remaining requantization or sampling boundaries are reported
in the export metadata.
