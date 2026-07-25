#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

source_checkpoint="artifacts/coat-pilot/coat_a4/checkpoint.pt"
projection="artifacts/coat/ternary-weights-projection.pt"
results="artifacts/attention-pilot/post-training-results.jsonl"

mkdir -p "$(dirname "${results}")"
: > "${results}"

evaluate_arm() {
  local scheme="$1"
  local threshold="$2"
  local clip="$3"
  uv run ternary-evaluate \
    --config configs/attention_pilot.toml \
    --checkpoint "${source_checkpoint}" \
    --mode coat_a4 \
    --projection "${projection}" \
    --batches 100 \
    --attention-quantization "${scheme}" \
    --attention-clip "${clip}" \
    --attention-threshold "${threshold}" \
    | tee -a "${results}"
}

train_arm() {
  local name="$1"
  local scheme="$2"
  local threshold="$3"
  local clip="$4"
  uv run ternary-train \
    --config configs/attention_pilot.toml \
    --init-from "${source_checkpoint}" \
    --projection "${projection}" \
    --run-name "${name}" \
    --attention-quantization "${scheme}" \
    --attention-clip "${clip}" \
    --attention-threshold "${threshold}"
}

evaluate_arm float 0.5 6.0
evaluate_arm score_int2 0.5 6.0
evaluate_arm score_int2 0.5 8.0
evaluate_arm score_int2 0.5 10.0
evaluate_arm score_int2 0.5 12.0
evaluate_arm prob_int2 0.5 6.0
evaluate_arm score_prob_int2 0.5 3.0
evaluate_arm score_prob_int2 0.5 6.0
evaluate_arm prob_binary 0.0625 6.0
evaluate_arm prob_binary 0.125 6.0
evaluate_arm prob_binary 0.25 6.0
evaluate_arm prob_binary 0.5 6.0

train_arm float float 0.5 6.0
train_arm score-int2-c8 score_int2 0.5 8.0
train_arm prob-int2 prob_int2 0.5 6.0
train_arm score-prob-int2-c3 score_prob_int2 0.5 3.0
train_arm binary-t0625 prob_binary 0.0625 6.0
train_arm binary-t125 prob_binary 0.125 6.0
