#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

config="configs/tinystories_28m.toml"
base="artifacts/tinystories-28m"
clip_experiment="${base}/attention-clip-refinement"
scale_experiment="${base}/shared-qkv-scale-refinement"
teacher_checkpoint="${base}/float-two-pass/checkpoint.pt"
experiment="${base}/softmax1-refinement"

selected_clip="$(jq -r '.selected_clip' "${clip_experiment}/selection.json")"
selected_initial="$(
  jq -r '.selected_initial_scale' "${scale_experiment}/selection.json"
)"
source_checkpoint="${base}/shared-head-qkv-scale-${selected_initial}-refine/checkpoint.pt"
if [[ ! -f "${source_checkpoint}" ]]; then
  echo "shared-head QKV scale checkpoint is missing" >&2
  exit 1
fi

mkdir -p "${experiment}"

for normalization in softmax softmax1; do
  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${source_checkpoint}" \
    --device cuda \
    --batches 200 \
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

  sha256sum "${run_dir}/checkpoint.pt" > "${run_dir}/SHA256SUMS"
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
