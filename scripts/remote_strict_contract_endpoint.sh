#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy
source scripts/remote_stage_helpers.sh

exec 9>/tmp/ternary-strict-contract-endpoint.lock
if ! flock -n 9; then
  echo "strict contract endpoint is already running" >&2
  exit 75
fi

config="configs/tinystories_28m.toml"
base="artifacts/tinystories-28m"
clip_experiment="${base}/attention-clip-refinement"
scale_experiment="${base}/shared-qkv-scale-refinement"
teacher_checkpoint="${base}/float-two-pass/checkpoint.pt"
experiment="${base}/strict-contract-refinement"
run_name="strict-contract-relu-rmsnorm"
run_dir="${base}/${run_name}"

if [[ -s "${experiment}/SUCCESS" ]]; then
  echo "strict contract endpoint is already complete: ${experiment}/SUCCESS"
  exit 0
fi
if [[ ! -s "${scale_experiment}/SUCCESS" ]]; then
  echo "shared QKV scale refinement must complete first" >&2
  exit 1
fi

selected_clip="$(jq -r '.selected_clip' "${clip_experiment}/selection.json")"
selected_initial="$(
  jq -r '.selected_initial_scale' "${scale_experiment}/selection.json"
)"
source_run="${base}/shared-head-qkv-scale-${selected_initial}-refine"
if [[ -s "${source_run}/best-checkpoint.pt" ]]; then
  source_checkpoint="${source_run}/best-checkpoint.pt"
elif [[ -s "${source_run}/checkpoint.pt" ]]; then
  source_checkpoint="${source_run}/checkpoint.pt"
else
  echo "shared-head QKV scale checkpoint is missing" >&2
  exit 1
fi

mkdir -p "${experiment}"
printf '%s\n' "${source_checkpoint}" > "${experiment}/source-checkpoint.txt"

# This is deliberately a strict branch, not the quality-winner handoff. It
# removes the three remaining operand-contract violations together: dynamic
# token scales, GELU, and floating RMSNorm. Training may retain shadow weights;
# the deployed operand path must use shared scales, ternary large operands,
# ReLU, and the exact integer RMSNorm reference.
if prepare_training_run "${run_dir}" 4000; then
  uv run ternary-train \
    --config "${config}" \
    --mode hadamard_progressive \
    --init-from "${source_checkpoint}" \
    --teacher-checkpoint "${teacher_checkpoint}" \
    --run-name "${run_name}" \
    --activation-encoding residual_ternary \
    --activation-planes 3 \
    --qkv-quantization ternary \
    --qkv-scale-granularity learned_head \
    --qkv-scale-initial "${selected_initial}" \
    --attention-quantization score_lut_prob_int2 \
    --attention-normalization softmax \
    --attention-clip "${selected_clip}" \
    --feed-forward-activation relu \
    --max-steps 4000 \
    --learning-rate 0.000012 \
    --min-learning-rate 0.000003 \
    --warmup-steps 100 \
    --weight-decay 0 \
    --logit-distillation-weight 0.1 \
    --attention-distillation-weight 0.5 \
    --hidden-distillation-weight 1.0
fi

uv run ternary-evaluate \
  --config "${config}" \
  --checkpoint "${run_dir}/checkpoint.pt" \
  --device cuda \
  --sequential \
  --rms-norm-quantization float \
  > "${run_dir}/full-validation-float-rmsnorm.json"

uv run ternary-evaluate \
  --config "${config}" \
  --checkpoint "${run_dir}/checkpoint.pt" \
  --device cuda \
  --sequential \
  --rms-norm-quantization integer_reference \
  > "${run_dir}/full-validation.json"

uv run ternary-diagnostics \
  --checkpoint "${run_dir}/checkpoint.pt" \
  --config "${config}" \
  --device cuda \
  --rms-norm-quantization integer_reference \
  > "${run_dir}/diagnostics.json"

: > "${run_dir}/generations.txt"
for prompt in \
  "Once upon a time" \
  "Lily found a tiny red door" \
  "Tom wanted to help his friend"
do
  uv run ternary-generate \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --tokenizer data/full/tokenizer.json \
    --prompt "${prompt}" \
    --max-new-tokens 120 \
    --temperature 0.8 \
    --top-k 50 \
    --device cuda \
    --rms-norm-quantization integer_reference \
    >> "${run_dir}/generations.txt"
done

uv run ternary-export \
  --checkpoint "${run_dir}/checkpoint.pt" \
  --output "${run_dir}/model-2bit.pt" \
  --rms-norm-quantization integer_reference \
  > "${run_dir}/packed-export.json"

uv run python - "${run_dir}" "${experiment}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
experiment = Path(sys.argv[2])
floating = json.loads((run / "full-validation-float-rmsnorm.json").read_text())
integer = json.loads((run / "full-validation.json").read_text())
packed = json.loads((run / "packed-export.json").read_text())
contract = packed["inference_contract"]
operand = contract["ternary_operand_contract"]
if not operand["satisfied"]:
    raise SystemExit(
        "strict endpoint failed ternary operand contract: "
        + ", ".join(operand["violations"])
    )
(experiment / "comparison.json").write_text(
    json.dumps(
        {
            "selection_rule": "strict ternary operand contract is mandatory",
            "run": run.name,
            "float_rmsnorm_loss": floating["loss"],
            "integer_rmsnorm_loss": integer["loss"],
            "integer_rmsnorm_delta": integer["loss"] - floating["loss"],
            "ternary_operand_contract_satisfied": True,
            "end_to_end_integer_reference_satisfied": contract[
                "end_to_end_integer_reference"
            ]["satisfied"],
            "remaining_integer_boundaries": contract[
                "end_to_end_integer_reference"
            ]["remaining_boundaries"],
        },
        indent=2,
    )
    + "\n"
)
PY

(
  cd "${run_dir}"
  checksum_files=(
    checkpoint.pt
    resolved-config.json
    metrics.jsonl
    full-validation.json
    full-validation-float-rmsnorm.json
    diagnostics.json
    generations.txt
    model-2bit.pt
    packed-export.json
  )
  if [[ -f best-checkpoint.pt ]]; then
    checksum_files+=(best-checkpoint.pt)
  fi
  sha256sum "${checksum_files[@]}" > SHA256SUMS
)
uv run ternary-audit-artifacts \
  --require-complete-checksums \
  "${run_dir}" \
  > "${run_dir}/artifact-audit.json"

printf 'run=%s\nselected_initial_scale=%s\nselected_clip=%s\n' \
  "${run_name}" "${selected_initial}" "${selected_clip}" \
  > "${experiment}/SUCCESS"
