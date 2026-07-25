# Research Run Log

## Full-corpus Stage A

- Started: 2026-07-25
- Dataset: `roneneldan/TinyStories`
- Train rows: 2,119,719
- Validation rows: 21,990
- Tokenizer: 4,096-token byte-level BPE trained on the fixed 50,000-story selection
- Model: 5,836,032 parameters, 6 layers, width 256, 8 heads, context 256
- Compared modes: `float`, `ternary_weights`, `ternary_activations`,
  `ternary_forward`
- Precision: BF16 forward/training arithmetic with FP32 optimizer parameters/state
- Full train tokens: 488,174,163
- Full validation tokens: 4,907,807
- Token budget: 488,177,664 sampled tokens per mode (29,796 steps × batch 64 ×
  context 256), one nominal full-corpus pass

### RunPod allocation

- Pod ID: `jf263l1xaj8aja`
- Name: `ternary-llm-stage-a`
- GPU: NVIDIA GeForce RTX 4090, 24 GB
- Host: Secure Cloud, Romania
- CPU/RAM: 16 vCPU / 61 GB
- Persistent pod volume: 40 GB mounted at `/workspace`
- Compute price at creation: $0.69/hour

The first US host allocation (`rviuel5z2wwmkf`) never reached container startup
(`uptimeSeconds` remained zero and no billing record appeared). It was deleted
before any files or training existed and replaced with the Romania allocation above.

### Gates

1. Bootstrap, CUDA verification, tests, and full data preparation.
2. Run 100-step BF16 benchmarks for `float` and `ternary_forward`.
3. Estimate time and cost for every Stage A mode from measured steady-state
   throughput.
4. Run the full floating-point baseline and sync its checkpoint locally.
5. Run the full ternary-weight ablation and sync its checkpoint locally.
6. Gate activation-heavy modes on population-coded residual pilots before spending
   a full pass.

The pod should be stopped if a correctness check fails or if the projected Stage A
compute cost exceeds $20 without a new decision.

### 100-step benchmark

| Mode | Median tokens/s | Estimated one-pass time | Estimated compute cost |
|---|---:|---:|---:|
| `float` | 245,032 | 0.55 h | $0.38 |
| `ternary_forward` | 552,234 | 0.25 h | $0.17 |

These are provisioning estimates, not quality results. The float validation loss
reached 5.216 after 1.64M tokens. Strict ternary reached 7.616 and showed very large
gradient norms during its first few steps, so all four modes must pass a longer
1,000-step warm-up pilot before full-corpus training begins.

### 1,000-step Stage A gate

Each mode processed 16,384,000 sampled tokens with the same seed, batches,
architecture, validation set, and 1,000-step warm-up schedule.

| Mode | Median tokens/s | Validation loss | Perplexity | Gate |
|---|---:|---:|---:|---|
| `float` | 1,156,353 | 3.215 | 24.91 | pass |
| `ternary_weights` | 1,109,368 | 3.572 | 35.58 | pass |
| `ternary_activations` | 564,921 | 6.053 | 425.54 | reject full pass |
| `ternary_forward` | 547,221 | 6.047 | 422.87 | reject full pass |

All four pilot checkpoints were copied to the local `artifacts/pilot` directory.
Only the two passing modes proceeded directly to a 488,177,664-token run.

Activation diagnostics locate a progressive residual-scale failure. In the
activation-only checkpoint, mean absolute residual magnitude grows from 3.88 after
block 0 to 66.91 after block 5, while the sign-change fraction between consecutive
blocks falls from 0.414 to 0.070. The float control's final block mean absolute
residual is 0.456. Output entropy is 6.58 for activation-only versus 3.31 for float,
consistent with an under-confident, information-poor predictor. Full diagnostic
JSON is stored beside each local pilot checkpoint.

### Later-milestone development gates

- Population residuals: deterministic 2- and 4-lane encoders pass exact-code,
  straight-through-gradient, model-gradient, and smoke-learning tests. At 100 smoke
  steps, P2 and P4 validation losses are 5.380 and 5.371.
- Counter training: scaled ternary parameters plus signed `int8` evidence counters
  use 125,524 optimizer-state bytes for the 119,104-parameter smoke model and learn
  in a 100-step development run, but remain oscillatory.
- Stochastic training: direct probabilistic code transitions retain only 6,420
  bytes of per-row scale state on the smoke model. Validation loss moves from about
  6.23 to 6.10 over 100 steps; this is a proof of learning signal, not a full-data
  readiness result.
- Packed inference: four codes are stored per byte. A 1024×1024 CPU reference
  compresses dense FP32 weight-plus-scale storage by 15.75× and matches dense
  outputs exactly. It is 4.02× slower because the portable reference unpacks before
  matrix multiplication.
- A first fused Triton CUDA kernel reads the packed codes inside the reduction
  without materializing dense weights. On the RTX 4090, a 256×1024 input and
  1024×1024 weight uses 266,240 packed-weight-plus-scale bytes versus 2,097,152
  BF16 dense bytes (7.88× smaller), but runs at 0.0755 ms versus cuBLAS BF16 at
  0.0203 ms. Packing and fusion are therefore complete as a reference, while the
  speed exit criterion remains unmet and requires kernel optimization.

### Full-pass results

| Mode | Sampled tokens | Validation loss | Perplexity | Local checkpoint |
|---|---:|---:|---:|---|
| `float` | 488,177,664 | 1.6989 | 5.47 | saved (67 MB) |
| `ternary_weights` | 488,177,664 | 2.0312 | 7.62 | saved (67 MB) |

The full float run completed without numerical failures. Its fixed-prompt generation
is coherent story text and is saved at
`artifacts/full-stage-a/float/sample.txt`.

The ternary-weight run also completed without numerical failures. Its final code
fractions are 32.18% `+1`, 32.43% `-1`, and 35.39% zero. The complete resumable
checkpoint and generated sample are local. Exporting its 39 forward tensors to
two-bit codes plus FP16 per-row scales produces a 1,521,077-byte inference artifact,
down from the 70,102,491-byte resumable checkpoint (which also contains shadow
weights and AdamW state).

### Activation-remediation results

| Run | Change from strict mode | Validation loss | Perplexity | Decision |
|---|---|---:|---:|---|
| P2 | 2 population lanes | 6.2206 | 503.01 | reject |
| P4 | 4 population lanes | 6.1606 | 473.72 | reject |
| strict-R0.25 | residual branch scale 0.25 | 6.0255 | 413.83 | reject |
| P4-R0.25 | P4 plus branch scale 0.25 | 7.3083 | 1492.59 | reject |

P2 and P4 show that adding deterministic representable levels alone does not rescue
the strict forward formulation. P4 controls residual magnitude better than the
single-lane pilot (final-block mean absolute residual 2.36), but output entropy
remains 6.85 and gradients spike. Branch scaling stabilizes ordinary strict-mode
gradients, but its residual sign-change rate falls below 0.006 through the middle
blocks, demonstrating frozen state rather than recovered capacity. A full-corpus
pass for these failed modes was therefore not run.

### Completion and local inventory

- The pod was stopped at 2026-07-25 11:48:24 UTC after all remote artifacts were
  copied and size-checked. It was then deleted after local verification to avoid
  idle-volume charges; `runpodctl pod list` returned an empty list. The remote copy
  is no longer recoverable. The allocation was active for about 50 minutes; at the
  quoted rate, continuous GPU compute for that window is approximately $0.58 before
  any provider-specific accounting adjustments.
- Local full data: 955 MB.
- Local experiment artifacts: 971 MB.
- Local checkpoints: 18 files totaling 848,249,859 bytes.
- Final validation: Ruff passes and 22 pytest tests pass.

Milestones 0–4 now have reproducible implementation and empirical results.
Milestones 5 and 6 have tiny-model learning prototypes and memory measurements but
do not meet their later full-scale exit criteria. Milestone 7 has a correct packed
format, portable reference, and fused Triton baseline; its speedup exit criterion
is explicitly unmet.

## COAT projection follow-up

- Date: 2026-07-25
- Paper: *COAT: COrrelation-Aware Orthogonal Transform for LLM Quantization*
- Source checkpoint: full-corpus `ternary_weights`, step 29,796
- Projection samples: 24,576 residual vectors
- Projection: closed-form `Q = U H` from streaming residual covariance
- Residual variance CV: 0.093699 before, `7.0e-9` after
- Maximum orthogonality error: `7.2e-7`

### Post-training activation ablation

Ten deterministic validation batches isolate the activation representation while
holding checkpoint weights fixed.

| Mode | Validation loss | Perplexity |
|---|---:|---:|
| `ternary_weights` (A16) | 2.0251 | 7.58 |
| `hadamard_a4` | 2.1712 | 8.77 |
| `coat_a4` | 2.1690 | 8.75 |
| `hadamard_ternary` | 8.6642 | 5,791.53 |
| `coat_ternary` | 8.6724 | 5,839.59 |

### Quantization-aware fine-tuning

Each projected mode was initialized from the same full-corpus ternary-weight
checkpoint and trained for 1,000 steps (16,384,000 sampled tokens) on the
tokenizer-compatible 50,000-story shard. Evaluation used 100 deterministic batches
from the full validation stream.

| Mode | Validation loss | Perplexity | Initial gradient norm |
|---|---:|---:|---:|
| `hadamard_a4` | 2.1156 | 8.29 | 0.876 |
| `coat_a4` | 2.1126 | 8.27 | 0.894 |
| `hadamard_ternary` | 6.3967 | 599.88 | `9.49e11` |
| `coat_ternary` | 6.3798 | 589.82 | `1.02e12` |

COAT is slightly better than the data-independent Hadamard control after
fine-tuning in both precisions. The much larger A4-versus-ternary gap shows that
perfect variance balancing does not by itself make a one-code residual stream
information-preserving.

The follow-up ran on RunPod pod `0ai5i7kndaqsbf`, an RTX 4090 at $0.69/hour.
The pod existed from 12:24:36 to approximately 12:50 UTC, including a 14-minute
one-time CUDA environment download. All four checkpoints and metrics were
size-checked locally before the pod was deleted. Approximate continuous allocation
cost was $0.30 before provider-specific accounting adjustments.
