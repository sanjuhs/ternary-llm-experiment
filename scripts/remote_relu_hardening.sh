#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy
source scripts/remote_stage_helpers.sh

exec 9>/tmp/ternary-relu-hardening.lock
if ! flock -n 9; then
  echo "ReLU hardening is already running" >&2
  exit 75
fi

config="configs/tinystories_28m.toml"
base="artifacts/tinystories-28m"
clip_experiment="${base}/attention-clip-refinement"
scale_experiment="${base}/shared-qkv-scale-refinement"
normalization_experiment="${base}/softmax1-refinement"
teacher_checkpoint="${base}/float-two-pass/checkpoint.pt"
experiment="${base}/relu-hardening"

if [[ -s "${experiment}/SUCCESS" ]]; then
  echo "ReLU hardening is already complete: ${experiment}/SUCCESS"
  exit 0
fi
if [[ ! -s "${normalization_experiment}/SUCCESS" ]]; then
  echo "Softmax1 refinement must complete before ReLU hardening" >&2
  exit 1
fi

selected_clip="$(jq -r '.selected_clip' "${clip_experiment}/selection.json")"
mkdir -p "${experiment}"

# Reject a refinement stage when both fixed-budget arms are worse than the
# checkpoint that entered it. Compare the original Softmax source and each
# arm's retained best checkpoint on identical batches.
normalization_source_checkpoint="$(
  <"${normalization_experiment}/source-checkpoint.txt"
)"
softmax_run="${base}/softmax-control-refine"
softmax1_run="${base}/softmax1-no-update-refine"
if [[ -s "${softmax_run}/best-checkpoint.pt" ]]; then
  softmax_checkpoint="${softmax_run}/best-checkpoint.pt"
elif [[ -s "${softmax_run}/checkpoint.pt" ]]; then
  softmax_checkpoint="${softmax_run}/checkpoint.pt"
else
  echo "matched Softmax checkpoint is missing" >&2
  exit 1
fi
if [[ -s "${softmax1_run}/best-checkpoint.pt" ]]; then
  softmax1_checkpoint="${softmax1_run}/best-checkpoint.pt"
elif [[ -s "${softmax1_run}/checkpoint.pt" ]]; then
  softmax1_checkpoint="${softmax1_run}/checkpoint.pt"
else
  echo "Softmax1 checkpoint is missing" >&2
  exit 1
fi

for declaration in \
  "step-zero:${normalization_source_checkpoint}" \
  "softmax-refined:${softmax_checkpoint}" \
  "softmax1-refined:${softmax1_checkpoint}"
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
  uv run python - \
    "${experiment}" \
    "${normalization_experiment}/source-selection.json" \
    "${normalization_source_checkpoint}" \
    "${softmax_checkpoint}" \
    "${softmax1_checkpoint}" <<'PY'
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
        "attention_normalization": "softmax",
    },
    "softmax_refined_best": {
        "checkpoint": sys.argv[4],
        "loss": json.loads(
            (root / "source-candidate-softmax-refined.json").read_text()
        )["loss"],
        "attention_normalization": "softmax",
    },
    "softmax1_refined_best": {
        "checkpoint": sys.argv[5],
        "loss": json.loads(
            (root / "source-candidate-softmax1-refined.json").read_text()
        )["loss"],
        "attention_normalization": "softmax1",
    },
}
winner = min(candidates, key=lambda name: candidates[name]["loss"])
selected = candidates[winner]
payload = {
    "selection_metric": "matched 200-batch validation loss",
    "candidates": candidates,
    "selected_candidate": winner,
    "selected_checkpoint": selected["checkpoint"],
    "selected_attention_normalization": selected["attention_normalization"],
    "selected_feed_forward_activation": "gelu",
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
selected_normalization="$(
  jq -r '.selected_attention_normalization' \
    "${experiment}/source-selection.json"
)"
selected_scale_granularity="$(
  jq -r '.selected_qkv_scale_granularity' \
    "${experiment}/source-selection.json"
)"
selected_scale_initial="$(
  jq -r '.selected_qkv_scale_initial // empty' \
    "${experiment}/source-selection.json"
)"
qkv_scale_args=(--qkv-scale-granularity "${selected_scale_granularity}")
if [[ "${selected_scale_granularity}" == "learned_head" ]]; then
  qkv_scale_args+=(--qkv-scale-initial "${selected_scale_initial}")
fi
printf '%s\n' "${source_checkpoint}" > "${experiment}/source-checkpoint.txt"

# ReLU removes GELU's tanh/polynomial approximation from the deployed FFN.
# Train an unchanged GELU arm for exactly the same budget so a difference
# cannot be attributed to extra optimization.
for activation in gelu relu; do
  if [[ "${activation}" == "gelu" ]]; then
    run_name="ffn-gelu-control-refine"
  else
    run_name="ffn-relu-harden-refine"
  fi

  if prepare_training_run "${base}/${run_name}" 4000; then
    uv run ternary-train \
      --config "${config}" \
      --mode hadamard_progressive \
      --init-from "${source_checkpoint}" \
      --teacher-checkpoint "${teacher_checkpoint}" \
      --run-name "${run_name}" \
      --activation-encoding residual_ternary \
      --activation-planes 3 \
      --qkv-quantization ternary \
      "${qkv_scale_args[@]}" \
      --attention-quantization score_lut_prob_int2 \
      --attention-normalization "${selected_normalization}" \
      --attention-clip "${selected_clip}" \
      --feed-forward-activation "${activation}" \
      --max-steps 4000 \
      --learning-rate 0.000012 \
      --min-learning-rate 0.000003 \
      --warmup-steps 100 \
      --weight-decay 0 \
      --logit-distillation-weight 0.1 \
      --attention-distillation-weight 0.5 \
      --hidden-distillation-weight 1.0
  fi

  run_dir="${base}/${run_name}"
  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --device cuda \
    --batches 200 \
    > "${experiment}/post-adaptation-${activation}.json"

  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --device cuda \
    --sequential \
    > "${run_dir}/full-validation.json"

  uv run ternary-diagnostics \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --config "${config}" \
    --device cuda \
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
      >> "${run_dir}/generations.txt"
  done

  uv run ternary-export \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --output "${run_dir}/model-2bit.pt" \
    > "${run_dir}/packed-export.json"

  (
    cd "${run_dir}"
    checksum_files=(
      checkpoint.pt \
      resolved-config.json \
      metrics.jsonl \
      full-validation.json \
      diagnostics.json \
      generations.txt \
      model-2bit.pt \
      packed-export.json
    )
    if [[ -f best-checkpoint.pt ]]; then
      checksum_files+=(best-checkpoint.pt)
    fi
    for preserved_checkpoint in checkpoint-step-*.pt; do
      if [[ -f "${preserved_checkpoint}" ]]; then
        checksum_files+=("${preserved_checkpoint}")
      fi
    done
    sha256sum "${checksum_files[@]}" > SHA256SUMS
  )
  uv run ternary-audit-artifacts \
    --require-complete-checksums \
    "${run_dir}" \
    > "${run_dir}/artifact-audit.json"
done

uv run python - "${experiment}" "${selected_normalization}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
normalization = sys.argv[2]
post = {
    activation: json.loads(
        (root / f"post-adaptation-{activation}.json").read_text()
    )["loss"]
    for activation in ("gelu", "relu")
}
full = {
    "gelu": json.loads(
        (
            root.parent
            / "ffn-gelu-control-refine"
            / "full-validation.json"
        ).read_text()
    )["loss"],
    "relu": json.loads(
        (
            root.parent
            / "ffn-relu-harden-refine"
            / "full-validation.json"
        ).read_text()
    )["loss"],
}
winner = min(full, key=full.get)
(root / "comparison.json").write_text(
    json.dumps(
        {
            "selection_metric": "matched exhaustive sequential validation loss",
            "source_attention_normalization": normalization,
            "post_adaptation_losses": post,
            "full_validation_losses": full,
            "relu_delta_vs_matched_gelu": full["relu"] - full["gelu"],
            "selected_feed_forward_activation": winner,
        },
        indent=2,
    )
    + "\n"
)
PY

printf 'source_checkpoint=%s\nselected_normalization=%s\n' \
  "${source_checkpoint}" "${selected_normalization}" > "${experiment}/SUCCESS"
