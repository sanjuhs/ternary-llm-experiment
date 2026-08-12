#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

exec 9>/tmp/ternary-llm-integer-rmsnorm-screen.lock
if ! flock -n 9; then
  echo "integer RMSNorm screen is already running" >&2
  exit 75
fi

base="artifacts/tinystories-28m"
config="configs/tinystories_28m.toml"
binary_experiment="${base}/binary-qk-fallback"
experiment="${base}/integer-rmsnorm-screen"

if [[ -s "${experiment}/SUCCESS" ]]; then
  echo "integer RMSNorm screen already complete"
  exit 0
fi
if [[ ! -s "${binary_experiment}/SUCCESS" ]]; then
  echo "binary Q/K fallback must complete before the integer RMSNorm screen" >&2
  exit 1
fi

mkdir -p "${experiment}"

# The binary-Q/K experiment may be rejected entirely. Compare its incoming
# ternary checkpoint with the retained best checkpoint from each matched arm.
binary_source_checkpoint="$(<"${binary_experiment}/source-checkpoint.txt")"
ternary_run="${base}/binary-qk-ternary-control-refine"
binary_run="${base}/binary-qk-sign-distill-refine"
if [[ -s "${ternary_run}/best-checkpoint.pt" ]]; then
  ternary_checkpoint="${ternary_run}/best-checkpoint.pt"
elif [[ -s "${ternary_run}/checkpoint.pt" ]]; then
  ternary_checkpoint="${ternary_run}/checkpoint.pt"
else
  echo "matched ternary-QKV checkpoint is missing" >&2
  exit 1
fi
if [[ -s "${binary_run}/best-checkpoint.pt" ]]; then
  binary_checkpoint="${binary_run}/best-checkpoint.pt"
elif [[ -s "${binary_run}/checkpoint.pt" ]]; then
  binary_checkpoint="${binary_run}/checkpoint.pt"
else
  echo "binary-Q/K checkpoint is missing" >&2
  exit 1
fi

for declaration in \
  "step-zero:${binary_source_checkpoint}" \
  "ternary-refined:${ternary_checkpoint}" \
  "binary-qk-refined:${binary_checkpoint}"
do
  IFS=: read -r candidate checkpoint <<< "${declaration}"
  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --device cuda \
    --batches 200 \
    > "${experiment}/source-candidate-${candidate}.json"
done

source_checkpoint="$(
  /opt/ternary-llm-venv/bin/python - \
    "${experiment}" \
    "${binary_experiment}/source-selection.json" \
    "${binary_source_checkpoint}" \
    "${ternary_checkpoint}" \
    "${binary_checkpoint}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
inherited = json.loads(Path(sys.argv[2]).read_text())
candidates = {
    "step_zero": {
        "checkpoint": sys.argv[3],
        "loss": json.loads(
            (root / "source-candidate-step-zero.json").read_text()
        )["loss"],
        "qkv_quantization": inherited["selected_qkv_quantization"],
    },
    "ternary_refined_best": {
        "checkpoint": sys.argv[4],
        "loss": json.loads(
            (root / "source-candidate-ternary-refined.json").read_text()
        )["loss"],
        "qkv_quantization": "ternary",
    },
    "binary_qk_refined_best": {
        "checkpoint": sys.argv[5],
        "loss": json.loads(
            (root / "source-candidate-binary-qk-refined.json").read_text()
        )["loss"],
        "qkv_quantization": "binary_qk_ternary_v",
    },
}
winner = min(candidates, key=lambda name: candidates[name]["loss"])
selected = candidates[winner]
payload = {
    "selection_metric": "matched 200-batch validation loss",
    "candidates": candidates,
    "selected_candidate": winner,
    "selected_checkpoint": selected["checkpoint"],
    "selected_qkv_quantization": selected["qkv_quantization"],
    "selected_feed_forward_activation": inherited[
        "selected_feed_forward_activation"
    ],
    "selected_attention_normalization": inherited[
        "selected_attention_normalization"
    ],
    "selected_qkv_scale_granularity": inherited[
        "selected_qkv_scale_granularity"
    ],
    "selected_qkv_scale_initial": inherited["selected_qkv_scale_initial"],
}
(root / "source-selection.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
print(selected["checkpoint"])
PY
)"
selected_qkv="$(
  jq -r '.selected_qkv_quantization' \
    "${experiment}/source-selection.json"
)"
printf '%s\n' "${source_checkpoint}" > "${experiment}/source-checkpoint.txt"

for normalization in float integer_reference; do
  uv run ternary-evaluate \
    --checkpoint "${source_checkpoint}" \
    --config "${config}" \
    --device cuda \
    --batches 200 \
    --rms-norm-quantization "${normalization}" \
    > "${experiment}/${normalization}-matched-200.json"
done

uv run python - "${experiment}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
matched = {
    name: json.loads((root / f"{name}-matched-200.json").read_text())
    for name in ("float", "integer_reference")
}
delta = matched["integer_reference"]["loss"] - matched["float"]["loss"]
(root / "matched-comparison.json").write_text(
    json.dumps(
        {
            "selection_metric": "matched 200-batch validation loss",
            "losses": {name: payload["loss"] for name, payload in matched.items()},
            "perplexities": {
                name: payload["perplexity"] for name, payload in matched.items()
            },
            "integer_rmsnorm_delta_vs_float": delta,
            "predeclared_practical_delta": 0.02,
            "passes_practical_screen": delta <= 0.02,
        },
        indent=2,
    )
    + "\n"
)
PY

for normalization in float integer_reference; do
  uv run ternary-evaluate \
    --checkpoint "${source_checkpoint}" \
    --config "${config}" \
    --device cuda \
    --sequential \
    --rms-norm-quantization "${normalization}" \
    > "${experiment}/${normalization}-full-validation.json"
done

uv run ternary-diagnostics \
  --checkpoint "${source_checkpoint}" \
  --config "${config}" \
  --device cuda \
  --rms-norm-quantization integer_reference \
  > "${experiment}/integer-reference-diagnostics.json"

: > "${experiment}/integer-reference-generations.txt"
for prompt in \
  "Once upon a time" \
  "Lily found a tiny red door" \
  "Tom wanted to help his friend"
do
  uv run ternary-generate \
    --checkpoint "${source_checkpoint}" \
    --tokenizer data/full/tokenizer.json \
    --prompt "${prompt}" \
    --max-new-tokens 120 \
    --temperature 0.8 \
    --top-k 50 \
    --device cuda \
    --rms-norm-quantization integer_reference \
    >> "${experiment}/integer-reference-generations.txt"
done

uv run ternary-export \
  --checkpoint "${source_checkpoint}" \
  --output "${experiment}/model-2bit-integer-rmsnorm.pt" \
  --rms-norm-quantization integer_reference \
  > "${experiment}/packed-export-integer-rmsnorm.json"

uv run python - "${experiment}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
full = {
    name: json.loads((root / f"{name}-full-validation.json").read_text())
    for name in ("float", "integer_reference")
}
delta = full["integer_reference"]["loss"] - full["float"]["loss"]
matched = json.loads((root / "matched-comparison.json").read_text())
(root / "comparison.json").write_text(
    json.dumps(
        {
            **matched,
            "final_comparison_metric": (
                "matched exhaustive sequential validation loss"
            ),
            "full_validation_losses": {
                name: payload["loss"] for name, payload in full.items()
            },
            "full_validation_perplexities": {
                name: payload["perplexity"] for name, payload in full.items()
            },
            "full_validation_integer_rmsnorm_delta_vs_float": delta,
            "passes_full_practical_delta": delta <= 0.02,
            "selected_rms_norm_quantization": (
                "integer_reference" if delta <= 0.02 else "float"
            ),
        },
        indent=2,
    )
    + "\n"
)
PY

(
  cd "${experiment}"
  sha256sum \
    comparison.json \
    float-full-validation.json \
    float-matched-200.json \
    integer-reference-diagnostics.json \
    integer_reference-full-validation.json \
    integer-reference-generations.txt \
    integer_reference-matched-200.json \
    matched-comparison.json \
    model-2bit-integer-rmsnorm.pt \
    packed-export-integer-rmsnorm.json \
    source-checkpoint.txt \
    > SHA256SUMS
  sha256sum --check SHA256SUMS
)

printf 'selected_qkv_quantization=%s\nsource_checkpoint=%s\n' \
  "${selected_qkv}" "${source_checkpoint}" > "${experiment}/SUCCESS"

# The currently running overnight orchestrator predates the separately declared
# strict-contract stage. Run it here as a compatibility bridge; newer
# orchestrators call the same idempotent script as the following stage.
bash scripts/remote_strict_contract_endpoint.sh
