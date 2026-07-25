#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

config="configs/residual_refinement_pilot.toml"
source_checkpoint="artifacts/fully-ternary-pilot/levels-7-gelu/checkpoint.pt"
teacher_checkpoint="artifacts/attention-pilot/prob-int2/checkpoint.pt"
projection="artifacts/coat/ternary-weights-projection.pt"
output="artifacts/residual-refinement-pilot"
screen="${output}/post-training-screen.jsonl"

mkdir -p "${output}"
: > "${screen}"

run_evaluation() {
  local mode=$1
  local encoding=$2
  local planes=$3
  local checkpoint=$4
  local projection_args=()
  if [[ "${mode}" == "coat_progressive" ]]; then
    projection_args=(--projection "${projection}")
  fi
  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --mode "${mode}" \
    "${projection_args[@]}" \
    --batches 100 \
    --activation-encoding "${encoding}" \
    --activation-planes "${planes}"
}

for mode in coat_progressive hadamard_progressive; do
  for encoding_planes in "residual_binary 2" "residual_ternary 2" "residual_ternary 3"; do
    read -r encoding planes <<< "${encoding_planes}"
    {
      printf '{"mode":"%s","encoding":"%s","planes":%s,"result":' \
        "${mode}" "${encoding}" "${planes}"
      run_evaluation "${mode}" "${encoding}" "${planes}" "${source_checkpoint}"
      printf '}\n'
    } >> "${screen}"
  done
done

train_arm() {
  local mode=$1
  local encoding=$2
  local planes=$3
  local run_name=$4
  local steps=$5
  local init_checkpoint=$6
  local projection_args=()
  if [[ "${mode}" == "coat_progressive" ]]; then
    projection_args=(--projection "${projection}")
  fi
  uv run ternary-train \
    --config "${config}" \
    --mode "${mode}" \
    --init-from "${init_checkpoint}" \
    "${projection_args[@]}" \
    --teacher-checkpoint "${teacher_checkpoint}" \
    --run-name "${run_name}" \
    --activation-encoding "${encoding}" \
    --activation-planes "${planes}" \
    --max-steps "${steps}"
}

train_arm coat_progressive residual_binary 2 coat-binary-p2 1500 "${source_checkpoint}"
train_arm coat_progressive residual_ternary 2 coat-ternary-p2 1500 "${source_checkpoint}"
train_arm coat_progressive residual_ternary 3 coat-ternary-p3 1500 "${source_checkpoint}"
train_arm hadamard_progressive residual_binary 2 hadamard-binary-p2 1500 \
  "${source_checkpoint}"
train_arm hadamard_progressive residual_ternary 2 hadamard-ternary-p2 1500 \
  "${source_checkpoint}"
train_arm hadamard_progressive residual_ternary 3 hadamard-ternary-p3 1500 \
  "${source_checkpoint}"

train_arm hadamard_progressive residual_binary 2 hadamard-binary-p2-long 5000 \
  "${output}/hadamard-binary-p2/checkpoint.pt"
train_arm hadamard_progressive residual_ternary 3 hadamard-ternary-p3-long 5000 \
  "${output}/hadamard-ternary-p3/checkpoint.pt"

for run_name in \
  coat-binary-p2 coat-ternary-p2 coat-ternary-p3 \
  hadamard-binary-p2 hadamard-ternary-p2 hadamard-ternary-p3 \
  hadamard-binary-p2-long hadamard-ternary-p3-long
do
  checkpoint="${output}/${run_name}/checkpoint.pt"
  uv run ternary-diagnostics \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
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
