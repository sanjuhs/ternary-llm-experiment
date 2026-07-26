#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

config="configs/tinystories_28m.toml"
base="artifacts/tinystories-28m"
clip_experiment="${base}/attention-clip-refinement"
teacher_checkpoint="${base}/float-two-pass/checkpoint.pt"
experiment="${base}/shared-qkv-scale-refinement"

if [[ -s "${experiment}/SUCCESS" ]]; then
  echo "shared QKV scale refinement is already complete: ${experiment}/SUCCESS"
  exit 0
fi

selected_clip="$(jq -r '.selected_clip' "${clip_experiment}/selection.json")"
source_run="${base}/attention-clip-selected-${selected_clip}-refine"
if [[ -s "${source_run}/best-checkpoint.pt" ]]; then
  source_checkpoint="${source_run}/best-checkpoint.pt"
elif [[ -s "${source_run}/checkpoint.pt" ]]; then
  source_checkpoint="${source_run}/checkpoint.pt"
else
  echo "selected clip-refinement checkpoint is missing" >&2
  exit 1
fi

mkdir -p "${experiment}"
printf '%s\n' "${source_checkpoint}" > "${experiment}/source-checkpoint.txt"

uv run ternary-evaluate \
  --config "${config}" \
  --checkpoint "${source_checkpoint}" \
  --device cuda \
  --batches 200 \
  > "${experiment}/pre-adaptation-token.json"

# A fixed learned scale per head makes the scale product factorizable outside
# both Q·K and Route·V. Screen the initialization before spending adaptation
# tokens because an initial scale that is too large can collapse every code to
# zero before the scale parameter receives a useful gradient.
for initial in 0.25 0.5 0.75 1.0; do
  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${source_checkpoint}" \
    --device cuda \
    --batches 200 \
    --qkv-scale-granularity learned_head \
    --qkv-scale-initial "${initial}" \
    > "${experiment}/pre-adaptation-initial-${initial}.json"
done

selected_initial="$(
  uv run python - "${experiment}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = {}
for path in sorted(root.glob("pre-adaptation-initial-*.json")):
    initial = path.stem.removeprefix("pre-adaptation-initial-")
    candidates[initial] = json.loads(path.read_text())["loss"]
winner = min(candidates, key=candidates.get)
print(winner)
(root / "selection.json").write_text(
    json.dumps(
        {
            "selection_metric": "matched 200-batch validation loss",
            "candidate_losses": candidates,
            "selected_initial_scale": winner,
        },
        indent=2,
    )
    + "\n"
)
PY
)"

# Match the learned-head arm with an unchanged per-token-scale control. This
# prevents four thousand extra optimizer steps from masquerading as a benefit
# of shared, factorizable scales.
control_run_name="qkv-scale-token-control"
uv run ternary-train \
  --config "${config}" \
  --mode hadamard_progressive \
  --init-from "${source_checkpoint}" \
  --teacher-checkpoint "${teacher_checkpoint}" \
  --run-name "${control_run_name}" \
  --activation-encoding residual_ternary \
  --activation-planes 3 \
  --qkv-quantization ternary \
  --qkv-scale-granularity token \
  --attention-quantization score_lut_prob_int2 \
  --attention-clip "${selected_clip}" \
  --max-steps 4000 \
  --learning-rate 0.000012 \
  --min-learning-rate 0.000003 \
  --warmup-steps 100 \
  --weight-decay 0 \
  --logit-distillation-weight 0.1 \
  --attention-distillation-weight 0.5 \
  --hidden-distillation-weight 1.0

learned_run_name="shared-head-qkv-scale-${selected_initial}-refine"
uv run ternary-train \
  --config "${config}" \
  --mode hadamard_progressive \
  --init-from "${source_checkpoint}" \
  --teacher-checkpoint "${teacher_checkpoint}" \
  --run-name "${learned_run_name}" \
  --activation-encoding residual_ternary \
  --activation-planes 3 \
  --qkv-quantization ternary \
  --qkv-scale-granularity learned_head \
  --qkv-scale-initial "${selected_initial}" \
  --attention-quantization score_lut_prob_int2 \
  --attention-clip "${selected_clip}" \
  --max-steps 4000 \
  --learning-rate 0.000012 \
  --min-learning-rate 0.000003 \
  --warmup-steps 100 \
  --weight-decay 0 \
  --logit-distillation-weight 0.1 \
  --attention-distillation-weight 0.5 \
  --hidden-distillation-weight 1.0

for arm in token learned_head; do
  if [[ "${arm}" == "token" ]]; then
    run_name="${control_run_name}"
  else
    run_name="${learned_run_name}"
  fi
  run_dir="${base}/${run_name}"

  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --device cuda \
    --batches 200 \
    > "${experiment}/post-adaptation-${arm}.json"

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

uv run python - "${experiment}" "${selected_initial}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
initial = sys.argv[2]
pre = {
    "token": json.loads((root / "pre-adaptation-token.json").read_text())["loss"],
    **{
        path.stem.removeprefix("pre-adaptation-initial-"): json.loads(
            path.read_text()
        )["loss"]
        for path in sorted(root.glob("pre-adaptation-initial-*.json"))
    },
}
post = {
    arm: json.loads((root / f"post-adaptation-{arm}.json").read_text())["loss"]
    for arm in ("token", "learned_head")
}
full = {
    "token": json.loads(
        (root.parent / "qkv-scale-token-control" / "full-validation.json").read_text()
    )["loss"],
    "learned_head": json.loads(
        (
            root.parent
            / f"shared-head-qkv-scale-{initial}-refine"
            / "full-validation.json"
        ).read_text()
    )["loss"],
}
(root / "comparison.json").write_text(
    json.dumps(
        {
            "initial_selection_metric": "matched 200-batch validation loss",
            "final_comparison_metric": (
                "matched exhaustive sequential validation loss"
            ),
            "selected_initial_scale": initial,
            "pre_adaptation_losses": pre,
            "post_adaptation_losses": post,
            "learned_head_delta_vs_matched_token": (
                post["learned_head"] - post["token"]
            ),
            "full_validation_losses": full,
            "full_validation_learned_head_delta_vs_matched_token": (
                full["learned_head"] - full["token"]
            ),
        },
        indent=2,
    )
    + "\n"
)
PY

printf 'selected_initial_scale=%s\ncontrol_run=%s\nlearned_run=%s\n' \
  "${selected_initial}" "${control_run_name}" "${learned_run_name}" \
  > "${experiment}/SUCCESS"
