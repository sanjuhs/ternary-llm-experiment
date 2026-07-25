#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

source_checkpoint="artifacts/gated-attention-pilot/lut-distilled-no-rect/checkpoint.pt"
teacher_checkpoint="artifacts/attention-pilot/prob-int2/checkpoint.pt"
projection="artifacts/coat/ternary-weights-projection.pt"
output="artifacts/fully-ternary-pilot"
ptq_results="${output}/post-training-level-screen.jsonl"

mkdir -p "${output}"
: > "${ptq_results}"

for levels in 19 15 11 7 3; do
  {
    printf '{"activation_levels":%s,"result":' "${levels}"
    uv run ternary-evaluate \
      --config configs/fully_ternary_pilot.toml \
      --checkpoint "${source_checkpoint}" \
      --mode coat_progressive \
      --projection "${projection}" \
      --batches 100 \
      --activation-levels "${levels}" \
      --qkv-quantization ternary \
      --attention-quantization score_lut_prob_int2 \
      --attention-clip 3.0 \
      --feed-forward-activation gelu
    printf '}\n'
  } >> "${ptq_results}"
done

previous="${source_checkpoint}"
for levels in 19 15 11 7 3; do
  run_name="levels-${levels}-gelu"
  uv run ternary-train \
    --config configs/fully_ternary_pilot.toml \
    --init-from "${previous}" \
    --projection "${projection}" \
    --teacher-checkpoint "${teacher_checkpoint}" \
    --run-name "${run_name}" \
    --activation-levels "${levels}" \
    --feed-forward-activation gelu
  previous="${output}/${run_name}/checkpoint.pt"
done

uv run ternary-train \
  --config configs/fully_ternary_pilot.toml \
  --init-from "${previous}" \
  --projection "${projection}" \
  --teacher-checkpoint "${teacher_checkpoint}" \
  --run-name levels-3-relu \
  --activation-levels 3 \
  --feed-forward-activation relu \
  --max-steps 750

relu_checkpoint="${output}/levels-3-relu/checkpoint.pt"
uv run ternary-train \
  --config configs/fully_ternary_pilot.toml \
  --init-from "${relu_checkpoint}" \
  --projection "${projection}" \
  --teacher-checkpoint "${teacher_checkpoint}" \
  --run-name levels-3-relu-binary-route \
  --activation-levels 3 \
  --feed-forward-activation relu \
  --attention-quantization score_lut_prob_binary \
  --attention-threshold 0.125 \
  --max-steps 750

for run_name in levels-3-gelu levels-3-relu levels-3-relu-binary-route; do
  checkpoint="${output}/${run_name}/checkpoint.pt"
  uv run ternary-diagnostics \
    --checkpoint "${checkpoint}" \
    --config configs/fully_ternary_pilot.toml \
    --device cuda \
    > "${output}/${run_name}/diagnostics.json"
  : > "${output}/${run_name}/generations.txt"
  for prompt in \
    "Once upon a time" \
    "Lily found a tiny red door" \
    "Tom wanted to help his friend"
  do
    uv run ternary-generate \
      --checkpoint "${checkpoint}" \
      --tokenizer data/processed/tokenizer.json \
      --prompt "${prompt}" \
      --max-new-tokens 120 \
      --temperature 0.8 \
      --top-k 50 \
      --device cuda \
      >> "${output}/${run_name}/generations.txt"
  done
done
