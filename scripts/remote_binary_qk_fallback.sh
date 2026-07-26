#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy
source scripts/remote_stage_helpers.sh

exec 9>/tmp/ternary-binary-qk-fallback.lock
if ! flock -n 9; then
  echo "binary Q/K fallback is already running" >&2
  exit 75
fi

config="configs/tinystories_28m.toml"
base="artifacts/tinystories-28m"
clip_experiment="${base}/attention-clip-refinement"
scale_experiment="${base}/shared-qkv-scale-refinement"
normalization_experiment="${base}/softmax1-refinement"
activation_experiment="${base}/relu-hardening"
teacher_checkpoint="${base}/float-two-pass/checkpoint.pt"
experiment="${base}/binary-qk-fallback"

if [[ -s "${experiment}/SUCCESS" ]]; then
  echo "Binary Q/K fallback is already complete: ${experiment}/SUCCESS"
  exit 0
fi
if [[ ! -s "${activation_experiment}/SUCCESS" ]]; then
  echo "ReLU hardening must complete before the binary Q/K fallback" >&2
  exit 1
fi

selected_clip="$(jq -r '.selected_clip' "${clip_experiment}/selection.json")"
selected_initial="$(
  jq -r '.selected_initial_scale' "${scale_experiment}/selection.json"
)"
selected_normalization="$(
  jq -r '.selected_normalization' "${normalization_experiment}/comparison.json"
)"
selected_activation="$(
  jq -r '.selected_feed_forward_activation' "${activation_experiment}/comparison.json"
)"

case "${selected_activation}" in
  gelu)
    source_name="ffn-gelu-control-refine"
    ;;
  relu)
    source_name="ffn-relu-harden-refine"
    ;;
  *)
    echo "unsupported selected feed-forward activation: ${selected_activation}" >&2
    exit 1
    ;;
esac
source_run="${base}/${source_name}"
if [[ -s "${source_run}/best-checkpoint.pt" ]]; then
  source_checkpoint="${source_run}/best-checkpoint.pt"
elif [[ -s "${source_run}/checkpoint.pt" ]]; then
  source_checkpoint="${source_run}/checkpoint.pt"
else
  echo "selected feed-forward checkpoint is missing" >&2
  exit 1
fi

mkdir -p "${experiment}"
printf '%s\n' "${source_checkpoint}" > "${experiment}/source-checkpoint.txt"

for qkv in ternary binary_qk_ternary_v; do
  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${source_checkpoint}" \
    --device cuda \
    --batches 200 \
    --qkv-quantization "${qkv}" \
    > "${experiment}/pre-adaptation-${qkv}.json"
done

# BinaryAttention's useful transferable idea is sign-preserving Q/K
# distillation. Both arms receive the same Q/K-similarity loss and optimizer
# budget, so the comparison isolates the Q/K alphabet rather than extra QAT.
for qkv in ternary binary_qk_ternary_v; do
  if [[ "${qkv}" == "ternary" ]]; then
    run_name="binary-qk-ternary-control-refine"
  else
    run_name="binary-qk-sign-distill-refine"
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
      --qkv-quantization "${qkv}" \
      --qkv-scale-granularity learned_head \
      --qkv-scale-initial "${selected_initial}" \
      --attention-quantization score_lut_prob_int2 \
      --attention-normalization "${selected_normalization}" \
      --attention-clip "${selected_clip}" \
      --feed-forward-activation "${selected_activation}" \
      --max-steps 4000 \
      --learning-rate 0.000012 \
      --min-learning-rate 0.000003 \
      --warmup-steps 100 \
      --weight-decay 0 \
      --logit-distillation-weight 0.1 \
      --attention-distillation-weight 0.5 \
      --qk-distillation-weight 0.5 \
      --hidden-distillation-weight 1.0
  fi

  run_dir="${base}/${run_name}"
  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --device cuda \
    --batches 200 \
    > "${experiment}/post-adaptation-${qkv}.json"

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
      checkpoint.pt
      resolved-config.json
      metrics.jsonl
      full-validation.json
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
done

uv run python - "${experiment}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = ("ternary", "binary_qk_ternary_v")
pre = {
    name: json.loads((root / f"pre-adaptation-{name}.json").read_text())["loss"]
    for name in names
}
post = {
    name: json.loads((root / f"post-adaptation-{name}.json").read_text())["loss"]
    for name in names
}
full = {
    "ternary": json.loads(
        (
            root.parent
            / "binary-qk-ternary-control-refine"
            / "full-validation.json"
        ).read_text()
    )["loss"],
    "binary_qk_ternary_v": json.loads(
        (
            root.parent
            / "binary-qk-sign-distill-refine"
            / "full-validation.json"
        ).read_text()
    )["loss"],
}
winner = min(full, key=full.get)
(root / "comparison.json").write_text(
    json.dumps(
        {
            "selection_metric": "matched exhaustive sequential validation loss",
            "pre_adaptation_losses": pre,
            "post_adaptation_losses": post,
            "full_validation_losses": full,
            "binary_qk_delta_vs_matched_ternary_qkv": (
                full["binary_qk_ternary_v"] - full["ternary"]
            ),
            "selected_qkv_quantization": winner,
        },
        indent=2,
    )
    + "\n"
)
PY

printf 'ternary_control=binary-qk-ternary-control-refine\nbinary_qk=binary-qk-sign-distill-refine\n' \
  > "${experiment}/SUCCESS"
