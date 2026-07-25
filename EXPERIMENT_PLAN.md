# Ternary Language Model Experiment

## 1. Research question

Can a small autoregressive language model learn useful language when its persistent
forward-pass representations use ternary codes `{-1, 0, +1}`, while reductions use
wider temporary accumulators?

The experiment separates three questions that are easy to conflate:

1. **Representation:** Are ternary weights and activations expressive enough?
2. **Optimization state:** Can high-precision shadow weights be removed?
3. **Learning rule:** Can numeric update counters also be replaced?

These questions must be tested in order. A failed result in a later stage should not
be mistaken for evidence that ternary forward representation itself is insufficient.

## 2. Working definition of “ternary”

The practical machine model is:

```text
ternary operand × ternary operand
    → integer or floating-point reduction
    → scale / normalize / threshold
    → ternary code
```

The model does **not** require accumulators, logits, loss values, gradients, or
optimizer state to take only three values. A dot product over `n` ternary pairs has
`2n + 1` possible exact sums, so restricting the accumulator itself to three values
would destroy evidence during the reduction.

For this project:

- A ternary tensor is represented as a three-valued code plus a scale.
- Matrix operands can be ternary even when the prototype uses ordinary PyTorch
  kernels to emulate the arithmetic.
- Reductions, normalization statistics, logits, and loss stay at normal precision.
- “Strict forward” means that learned weights and persistent hidden states are
  requantized at the boundaries documented below.
- Runtime speedup is not claimed until packed, fused low-bit kernels exist.

## 3. Hypotheses

### Primary hypothesis

A 5–15 million parameter decoder-only transformer trained on TinyStories can learn
coherent short-form language with ternary learned weights and aggressively
ternarized hidden states.

### Secondary hypotheses

- Weight ternarization should cost less quality than activation and residual-stream
  ternarization.
- Repeated residual-stream ternarization will be the main quality bottleneck.
- Population-coded residual lanes should preserve small updates better than one
  ternary value per feature.
- Removing high-precision shadow weights will destabilize training unless update
  evidence is accumulated over time.

## 4. Experimental controls

All comparable runs must use the same:

- TinyStories train/validation samples
- tokenizer and vocabulary
- model width, depth, context length, and parameter budget
- optimizer schedule and number of processed tokens
- random seed, except for explicitly repeated-seed runs
- evaluation batches and generation prompts

Every run records its resolved configuration, parameter count, token count, losses,
perplexity, throughput, checkpoint size, and sample generations.

## 5. Model

The first model is a small GPT-style decoder:

- causal self-attention
- pre-normalization transformer blocks
- gated or GELU feed-forward network
- learned token and position embeddings
- tied token embedding and language-model head where practical
- configurable context length, width, heads, layers, and dropout

The default research configuration targets roughly 10 million parameters so it can
run on a laptop. A much smaller smoke configuration is used by tests.

### Forward modes

| Mode | Learned weights | Hidden activations | Residual stream | Purpose |
|---|---|---|---|---|
| `float` | floating point | floating point | floating point | quality and implementation control |
| `ternary_weights` | fake-quantized ternary | floating point | floating point | isolate weight quantization |
| `ternary_activations` | floating point | ternary at block boundaries | ternary | isolate activation pressure |
| `ternary_forward` | fake-quantized ternary | ternary at documented boundaries | ternary | Stage A strict-forward test |
| `population_ternary` | fake-quantized ternary | ternary lanes | population-coded lanes | residual-capacity test |
| `hadamard_a4` | fake-quantized ternary | asymmetric A4 | Hadamard-projected A4 | rotation control |
| `coat_a4` | fake-quantized ternary | asymmetric A4 | COAT-projected A4 | positive low-bit control |
| `hadamard_ternary` | fake-quantized ternary | ternary | Hadamard-projected ternary | rotation control |
| `coat_ternary` | fake-quantized ternary | ternary | COAT-projected ternary | experimental COAT extension |

The prototype uses deterministic thermometer-style population coding. For `L`
lanes, each normalized scalar is compared with `L` evenly spaced magnitude
thresholds, producing exact per-lane codes in `{-1, 0, +1}`. The temporary decoded
value is the mean lane code times the detached per-token scale. At the default
threshold, one lane is identical to the ordinary activation ternarizer; two and
four lanes add progressively finer levels without adding learned parameters.

Fake quantization uses the straight-through estimator during training. The forward
value is quantized; the backward pass supplies a usable approximate gradient.

The projected modes rotate the persistent residual state into an orthogonal basis,
quantize there, and decode for the ordinary reference kernels. `hadamard_*` uses a
fixed normalized Sylvester matrix. `coat_*` requires a calibration artifact whose
projection is computed as `Q = U H` from a streaming residual covariance estimate.
The A4 arm uses 16 asymmetric per-token levels. It is a matched control for whether
the projection works at the precision evaluated by the COAT paper; COAT itself does
not establish ternary activations.

### Ternary weight rule

For a weight group `w`, calculate a positive scale `α`, quantize `w / α` to
`{-1, 0, +1}`, and use `αq` in the forward operation. The default grouping is per
output channel for linear layers and per embedding row for embeddings. The exact
scale rule and zero threshold are configuration values and are logged.

### Ternary activation rule

For an activation group `x`, calculate a detached scale, clamp and round the
normalized values to `{-1, 0, +1}`, and pass the scaled code to the next operation.
The default grouping is per token. Quantization is applied:

- after token-plus-position embedding
- to the inputs of quantized linear operations
- after attention and feed-forward branch outputs
- after each residual addition in strict-forward mode
- before the language-model projection

Attention logits and softmax remain floating point in Stage A. Q, K, and V operands
may be ternary, but their dot-product reductions are wider values.

### Normalization

Normalization statistics are computed in floating point. In strict-forward mode,
the affine parameters are ternarized and the normalized output is requantized before
it becomes the next persistent hidden state.

## 6. Dataset and tokenizer

Source dataset: [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)

The data workflow will:

1. verify the dataset and resolve its available splits;
2. download it through Hugging Face Datasets into a local cache;
3. create deterministic train and validation selections;
4. train or load a reproducible byte-level BPE tokenizer;
5. tokenize stories with explicit beginning/end markers;
6. write contiguous token shards plus metadata for memory-efficient training.

Raw downloads, generated shards, and checkpoints are local artifacts and are not
committed to source control. A bounded sample is the default for the first runnable
experiment; the full split is opt-in so smoke validation does not unexpectedly
consume hours.

## 7. Metrics

### Required

- validation cross-entropy
- validation perplexity
- training tokens and optimizer steps
- tokens per second
- model parameter count
- serialized checkpoint size
- fraction of quantized weight codes equal to `-1`, `0`, and `+1`
- fraction of activation codes equal to `-1`, `0`, and `+1`
- generated samples from a fixed prompt suite

### Diagnostic

- gradient norm
- activation scale by layer
- saturation and zero rate by layer
- attention entropy
- residual code-change rate
- peak device memory when available

Quality comparisons are valid only at matched processed-token budgets.

## 8. Experiment sequence

### Milestone 0 — Reproducible project

- initialize a `uv` Python project;
- pin supported Python and dependencies;
- add formatting, linting, type checking, and tests;
- add deterministic configuration and device selection;
- prepare TinyStories and the tokenizer;
- make a tiny overfit/smoke run pass.

Exit criterion: a fresh checkout can run tests, prepare a small sample, train for a
few steps, evaluate, save a checkpoint, reload it, and generate text.

### Milestone 1 — Floating-point baseline

Train the default model in `float` mode. Confirm that loss falls, generated samples
improve, checkpoint resume is exact enough for continued training, and validation
perplexity is reproducible.

Exit criterion: the baseline completes the agreed token budget without numerical
failure and produces coherent TinyStories-style text.

### Milestone 2 — Ternary weights

Run `ternary_weights` at the same architecture and token budget. Measure quality,
code balance, and optimization stability relative to the baseline.

Exit criterion: training is stable and the quality delta is quantified.

### Milestone 3 — Strict ternary forward representation

Run the activation-only ablation, then `ternary_forward`. Ternarize embeddings,
linear operands, branch outputs, residual states, normalization affine parameters,
and the output projection operands. Keep reductions and the training machinery at
normal precision.

Exit criterion: determine whether the strict forward representation learns, and
identify the first layer or boundary where information collapses if it does not.

### Milestone 4 — Population-coded residuals

Represent each logical residual feature with multiple ternary lanes. Combine lanes
only for wider temporary computation, then requantize each lane.

Compare one, two, and four lanes at matched total parameter or compute budgets.

Exit criterion: determine whether lanes improve validation loss or residual
code-change rate enough to justify their added width.

### Milestone 5 — Remove shadow weights

This is Stage B from the original idea. Store ternary model weights and a small
signed update counter for each parameter. Accumulate update evidence until a
threshold causes `-1 → 0 → +1` or the reverse.

Begin with tiny models and overfit tests before TinyStories-scale training.

Exit criterion: demonstrate learning without persistent floating-point shadow
weights and compare total training-state memory with AdamW.

### Milestone 6 — Stochastic transitions

This is Stage C. Replace numeric counters with probabilistic state transitions whose
rates depend on update evidence. Precision exists statistically over repeated
events, not as a stored high-bit scalar.

Exit criterion: establish whether the method learns algorithmic tasks and a tiny
character-level language model before attempting the main dataset.

### Milestone 7 — Packed inference kernels

Only after a model is worth accelerating:

- pack ternary codes into two-bit slots;
- implement add/subtract/skip reductions;
- fuse scale, bias, activation, and requantization;
- benchmark against PyTorch floating-point and fake-quantized implementations.

Exit criterion: report real latency, throughput, memory bandwidth, and energy
measurements without equating model size reduction with speedup.

## 9. Run matrix

The minimum defensible matrix is:

| Run | Mode | Seeds | Purpose |
|---|---|---:|---|
| S0 | tiny smoke | 1 | correctness |
| B0 | `float` | 3 | baseline variance |
| W0 | `ternary_weights` | 3 | weight effect |
| A0 | `ternary_activations` | 3 | activation/residual effect |
| T0 | `ternary_forward` | 3 | combined Stage A result |
| P2 | `population_ternary`, 2 lanes | 3 | residual coding |
| P4 | `population_ternary`, 4 lanes | 3 | residual coding |

Development runs may be shorter, but final comparisons use matched token budgets.

## 10. Repository layout

```text
.
├── EXPERIMENT_PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── smoke.toml
│   └── tiny.toml
├── scripts/
│   └── run_stage_a.sh
├── src/ternary_llm/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── quantization.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
└── tests/
```

Local-only directories:

```text
data/
artifacts/
.cache/
```

## 11. Reproducibility rules

- All commands run through `uv run`.
- Config files are immutable inputs; the resolved config is copied into each run.
- Seeds cover Python, NumPy, and PyTorch.
- Dataset selections are deterministic and recorded by source revision when
  available.
- Checkpoints contain model state, optimizer state, scheduler state, step, processed
  tokens, configuration, tokenizer reference, and random-number-generator states.
- Tests never require the full dataset or a GPU.
- CPU and Apple Metal (`mps`) are supported first; CUDA remains a compatible target.

## 12. Risks and decision rules

- **Loss does not fall in smoke overfit:** treat as an implementation bug.
- **Weight-only mode fails:** inspect scale, threshold, initialization, and STE
  before testing stricter modes.
- **Residual codes freeze or oscillate:** test learned thresholds, branch scaling,
  and population lanes.
- **Softmax becomes unstable:** keep logits and softmax in floating point; this does
  not violate the operand-focused hypothesis.
- **Fake quantization is slow:** accept it for research correctness; kernel work is a
  later milestone.
- **A larger model is requested:** scale only after the tiny configuration is
  reproducible and the ablation matrix is affordable.

## 13. Immediate implementation target

This setup pass implements Milestone 0 and the machinery needed for Milestones 1–3:

- environment and dependencies;
- dataset inspection, download, selection, tokenizer, and token shards;
- configurable transformer and ternary fake-quantization primitives;
- training, evaluation, checkpoint, resume, and generation commands;
- tests and a bounded end-to-end smoke run.

Long baseline training, multi-seed comparison, discrete counter optimization,
stochastic learning, and custom kernels remain explicit later experiments rather
than being silently simulated.
