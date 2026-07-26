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

selected_clip="$(jq -r '.selected_clip' "${clip_experiment}/selection.json")"
source_checkpoint="$(
  find "${base}" -maxdepth 2 -type f \
    -path "${base}/attention-clip-selected-${selected_clip}-refine/checkpoint.pt" \
    -print -quit
)"
if [[ -z "${source_checkpoint}" ]]; then
  echo "selected clip-refinement checkpoint is missing" >&2
  exit 1
fi

mkdir -p "${experiment}"

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

run_name="shared-head-qkv-scale-${selected_initial}-refine"
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
  --attention-clip "${selected_clip}" \
  --max-steps 4000 \
  --learning-rate 0.000012 \
  --min-learning-rate 0.000003 \
  --warmup-steps 100 \
  --weight-decay 0 \
  --logit-distillation-weight 0.1 \
  --attention-distillation-weight 0.5 \
  --hidden-distillation-weight 1.0

run_dir="${base}/${run_name}"
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
    --tokenizer data/processed/tokenizer.json \
    --prompt "${prompt}" \
    --max-new-tokens 120 \
    --temperature 0.8 \
    --top-k 50 \
    --device cuda \
    >> "${run_dir}/generations.txt"
done

sha256sum "${run_dir}/checkpoint.pt" > "${run_dir}/SHA256SUMS"
