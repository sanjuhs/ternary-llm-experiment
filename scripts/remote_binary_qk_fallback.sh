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
mkdir -p "${experiment}"

# Compare the ReLU stage's incoming checkpoint with each arm's retained best
# checkpoint. This lets the architecture change be rejected without losing a
# better upstream model.
activation_source_checkpoint="$(<"${activation_experiment}/source-checkpoint.txt")"
gelu_run="${base}/ffn-gelu-control-refine"
relu_run="${base}/ffn-relu-harden-refine"
if [[ -s "${gelu_run}/best-checkpoint.pt" ]]; then
  gelu_checkpoint="${gelu_run}/best-checkpoint.pt"
elif [[ -s "${gelu_run}/checkpoint.pt" ]]; then
  gelu_checkpoint="${gelu_run}/checkpoint.pt"
else
  echo "matched GELU checkpoint is missing" >&2
  exit 1
fi
if [[ -s "${relu_run}/best-checkpoint.pt" ]]; then
  relu_checkpoint="${relu_run}/best-checkpoint.pt"
elif [[ -s "${relu_run}/checkpoint.pt" ]]; then
  relu_checkpoint="${relu_run}/checkpoint.pt"
else
  echo "ReLU checkpoint is missing" >&2
  exit 1
fi

for declaration in \
  "step-zero:${activation_source_checkpoint}" \
  "gelu-refined:${gelu_checkpoint}" \
  "relu-refined:${relu_checkpoint}"
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
    "${activation_experiment}/source-selection.json" \
    "${activation_source_checkpoint}" \
    "${gelu_checkpoint}" \
    "${relu_checkpoint}" <<'PY'
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
        "feed_forward_activation": inherited[
            "selected_feed_forward_activation"
        ],
    },
    "gelu_refined_best": {
        "checkpoint": sys.argv[4],
        "loss": json.loads(
            (root / "source-candidate-gelu-refined.json").read_text()
        )["loss"],
        "feed_forward_activation": "gelu",
    },
    "relu_refined_best": {
        "checkpoint": sys.argv[5],
        "loss": json.loads(
            (root / "source-candidate-relu-refined.json").read_text()
        )["loss"],
        "feed_forward_activation": "relu",
    },
}
winner = min(candidates, key=lambda name: candidates[name]["loss"])
selected = candidates[winner]
payload = {
    "selection_metric": "matched 200-batch validation loss",
    "candidates": candidates,
    "selected_candidate": winner,
    "selected_checkpoint": selected["checkpoint"],
    "selected_qkv_quantization": "ternary",
    "selected_feed_forward_activation": selected[
        "feed_forward_activation"
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
selected_activation="$(
  jq -r '.selected_feed_forward_activation' \
    "${experiment}/source-selection.json"
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
      "${qkv_scale_args[@]}" \
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
