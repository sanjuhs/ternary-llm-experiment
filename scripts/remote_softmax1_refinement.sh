#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy
source scripts/remote_stage_helpers.sh

exec 9>/tmp/ternary-softmax1-refinement.lock
if ! flock -n 9; then
  echo "Softmax1 refinement is already running" >&2
  exit 75
fi

config="configs/tinystories_28m.toml"
base="artifacts/tinystories-28m"
clip_experiment="${base}/attention-clip-refinement"
scale_experiment="${base}/shared-qkv-scale-refinement"
teacher_checkpoint="${base}/float-two-pass/checkpoint.pt"
experiment="${base}/softmax1-refinement"

if [[ -s "${experiment}/SUCCESS" ]]; then
  echo "Softmax1 refinement is already complete: ${experiment}/SUCCESS"
  exit 0
fi
if [[ ! -s "${scale_experiment}/SUCCESS" ]]; then
  echo "shared QKV scale refinement must complete before Softmax1 refinement" >&2
  exit 1
fi

selected_clip="$(jq -r '.selected_clip' "${clip_experiment}/selection.json")"
selected_initial="$(
  jq -r '.selected_initial_scale' "${scale_experiment}/selection.json"
)"
mkdir -p "${experiment}"

# The shared-scale stage reports fixed-budget endpoints, but either trained arm
# may be worse than the step-zero clip source. Re-evaluate all deployable
# checkpoints identically before selecting the source for this stage.
scale_source_checkpoint="$(<"${scale_experiment}/source-checkpoint.txt")"
token_run="${base}/qkv-scale-token-control"
learned_run="${base}/shared-head-qkv-scale-${selected_initial}-refine"
if [[ -s "${token_run}/best-checkpoint.pt" ]]; then
  token_checkpoint="${token_run}/best-checkpoint.pt"
elif [[ -s "${token_run}/checkpoint.pt" ]]; then
  token_checkpoint="${token_run}/checkpoint.pt"
else
  echo "matched token-scale checkpoint is missing" >&2
  exit 1
fi
if [[ -s "${learned_run}/best-checkpoint.pt" ]]; then
  learned_checkpoint="${learned_run}/best-checkpoint.pt"
elif [[ -s "${learned_run}/checkpoint.pt" ]]; then
  learned_checkpoint="${learned_run}/checkpoint.pt"
else
  echo "shared-head QKV scale checkpoint is missing" >&2
  exit 1
fi

uv run ternary-evaluate \
  --config "${config}" \
  --checkpoint "${scale_source_checkpoint}" \
  --device cuda \
  --batches 200 \
  > "${experiment}/source-candidate-step-zero.json"
uv run ternary-evaluate \
  --config "${config}" \
  --checkpoint "${token_checkpoint}" \
  --device cuda \
  --batches 200 \
  > "${experiment}/source-candidate-token-refined.json"
uv run ternary-evaluate \
  --config "${config}" \
  --checkpoint "${learned_checkpoint}" \
  --device cuda \
  --batches 200 \
  > "${experiment}/source-candidate-learned-head-refined.json"

source_checkpoint="$(
  uv run python - \
    "${experiment}" \
    "${scale_source_checkpoint}" \
    "${token_checkpoint}" \
    "${learned_checkpoint}" \
    "${selected_initial}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = {
    "step_zero": {
        "checkpoint": sys.argv[2],
        "loss": json.loads(
            (root / "source-candidate-step-zero.json").read_text()
        )["loss"],
        "qkv_scale_granularity": "token",
        "qkv_scale_initial": None,
    },
    "token_refined_best": {
        "checkpoint": sys.argv[3],
        "loss": json.loads(
            (root / "source-candidate-token-refined.json").read_text()
        )["loss"],
        "qkv_scale_granularity": "token",
        "qkv_scale_initial": None,
    },
    "learned_head_refined_best": {
        "checkpoint": sys.argv[4],
        "loss": json.loads(
            (root / "source-candidate-learned-head-refined.json").read_text()
        )["loss"],
        "qkv_scale_granularity": "learned_head",
        "qkv_scale_initial": float(sys.argv[5]),
    },
}
winner = min(candidates, key=lambda name: candidates[name]["loss"])
selected = candidates[winner]
(root / "source-selection.json").write_text(
    json.dumps(
        {
            "selection_metric": "matched 200-batch validation loss",
            "candidates": candidates,
            "selected_candidate": winner,
            "selected_checkpoint": selected["checkpoint"],
            "selected_qkv_scale_granularity": selected[
                "qkv_scale_granularity"
            ],
            "selected_qkv_scale_initial": selected["qkv_scale_initial"],
        },
        indent=2,
    )
    + "\n"
)
print(selected["checkpoint"])
PY
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

for normalization in softmax softmax1; do
  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${source_checkpoint}" \
    --device cuda \
    --batches 200 \
    "${qkv_scale_args[@]}" \
    --attention-normalization "${normalization}" \
    > "${experiment}/pre-adaptation-${normalization}.json"
done

# Both arms start from the same shared-scale checkpoint and receive the same
# optimizer budget. Otherwise Softmax-1 would be confounded with additional
# adaptation.
for normalization in softmax softmax1; do
  if [[ "${normalization}" == "softmax" ]]; then
    run_name="softmax-control-refine"
  else
    run_name="softmax1-no-update-refine"
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
      --attention-normalization "${normalization}" \
      --attention-clip "${selected_clip}" \
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
    > "${experiment}/post-adaptation-${normalization}.json"

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

uv run python - "${experiment}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
pre = {
    name: json.loads((root / f"pre-adaptation-{name}.json").read_text())["loss"]
    for name in ("softmax", "softmax1")
}
post = {
    name: json.loads((root / f"post-adaptation-{name}.json").read_text())["loss"]
    for name in ("softmax", "softmax1")
}
full = {
    "softmax": json.loads(
        (root.parent / "softmax-control-refine" / "full-validation.json").read_text()
    )["loss"],
    "softmax1": json.loads(
        (
            root.parent
            / "softmax1-no-update-refine"
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
            "softmax1_delta_vs_matched_softmax": (
                post["softmax1"] - post["softmax"]
            ),
            "full_validation_losses": full,
            "full_validation_softmax1_delta_vs_matched_softmax": (
                full["softmax1"] - full["softmax"]
            ),
            "selected_normalization": winner,
        },
        indent=2,
    )
    + "\n"
)
PY

printf 'control_run=softmax-control-refine\nsoftmax1_run=softmax1-no-update-refine\n' \
  > "${experiment}/SUCCESS"
