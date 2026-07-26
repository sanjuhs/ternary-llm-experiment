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
source_checkpoint="$(
  find "${base}" -maxdepth 2 -type f \
    -path "${base}/shared-head-qkv-scale-*-refine/checkpoint.pt" \
    -print -quit
)"
if [[ -z "${source_checkpoint}" ]]; then
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

run_name="softmax1-no-update-refine"
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
  --attention-normalization softmax1 \
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
    --tokenizer data/full/tokenizer.json \
    --prompt "${prompt}" \
    --max-new-tokens 120 \
    --temperature 0.8 \
    --top-k 50 \
    --device cuda \
    >> "${run_dir}/generations.txt"
done

sha256sum "${run_dir}/checkpoint.pt" > "${run_dir}/SHA256SUMS"
