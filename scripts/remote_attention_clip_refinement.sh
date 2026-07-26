#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

config="configs/tinystories_28m.toml"
base="artifacts/tinystories-28m"
source_checkpoint="${base}/hadamard-ternary-p3-half-pass/checkpoint.pt"
teacher_checkpoint="${base}/float-two-pass/checkpoint.pt"
experiment="${base}/attention-clip-refinement"

mkdir -p "${experiment}"

# The clip=3 integer LUT produces exp(-3), exp(-2), exp(-1), exp(0).
# After row-wise 2-bit probability quantization these map to codes 0, 0, 1, 3,
# so code 2 is unused. Clips 1.5, 2.0, and 2.5 test progressively denser
# four-code routing under the same strict ternary model and teacher.
for clip in 1.5 2.0 2.5 3.0; do
  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${source_checkpoint}" \
    --device cuda \
    --batches 200 \
    --attention-clip "${clip}" \
    > "${experiment}/pre-adaptation-clip-${clip}.json"
done

for clip in 1.5 2.0 2.5; do
  run_name="attention-clip-${clip}-screen"
  uv run ternary-train \
    --config "${config}" \
    --mode hadamard_progressive \
    --init-from "${source_checkpoint}" \
    --teacher-checkpoint "${teacher_checkpoint}" \
    --run-name "${run_name}" \
    --activation-encoding residual_ternary \
    --activation-planes 3 \
    --qkv-quantization ternary \
    --attention-quantization score_lut_prob_int2 \
    --attention-clip "${clip}" \
    --max-steps 2000 \
    --learning-rate 0.00002 \
    --min-learning-rate 0.000003 \
    --warmup-steps 100 \
    --weight-decay 0 \
    --logit-distillation-weight 0.1 \
    --attention-distillation-weight 0.5 \
    --hidden-distillation-weight 1.0

  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${base}/${run_name}/checkpoint.pt" \
    --device cuda \
    --batches 200 \
    > "${experiment}/post-adaptation-clip-${clip}.json"
done

selection="$(
  uv run python - "${experiment}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = {}
for path in sorted(root.glob("post-adaptation-clip-*.json")):
    clip = path.stem.removeprefix("post-adaptation-clip-")
    candidates[clip] = json.loads(path.read_text())["loss"]
# The source checkpoint is already the fully adapted clip-3 baseline, so compare
# it directly rather than spending another redundant 2,000-step screen on it.
candidates["3.0"] = json.loads(
    (root / "pre-adaptation-clip-3.0.json").read_text()
)["loss"]
winner = min(candidates, key=candidates.get)
print(winner)
(root / "selection.json").write_text(
    json.dumps(
        {
            "selection_metric": "matched 200-batch validation loss",
            "candidate_losses": candidates,
            "selected_clip": winner,
        },
        indent=2,
    )
    + "\n"
)
PY
)"

if [[ "${selection}" == "3.0" ]]; then
  selected_checkpoint="${source_checkpoint}"
else
  selected_checkpoint="${base}/attention-clip-${selection}-screen/checkpoint.pt"
fi
run_name="attention-clip-selected-${selection}-refine"

uv run ternary-train \
  --config "${config}" \
  --mode hadamard_progressive \
  --init-from "${selected_checkpoint}" \
  --teacher-checkpoint "${teacher_checkpoint}" \
  --run-name "${run_name}" \
  --activation-encoding residual_ternary \
  --activation-planes 3 \
  --qkv-quantization ternary \
  --attention-quantization score_lut_prob_int2 \
  --attention-clip "${selection}" \
  --max-steps 5000 \
  --learning-rate 0.000015 \
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
