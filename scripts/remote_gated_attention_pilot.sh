#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

source_checkpoint="artifacts/attention-pilot/prob-int2/checkpoint.pt"
projection="artifacts/coat/ternary-weights-projection.pt"
results="artifacts/gated-attention-pilot/post-training-results.jsonl"

mkdir -p "$(dirname "${results}")"
: > "${results}"

evaluate_arm() {
  local name="$1"
  local qkv="$2"
  local rectification="$3"
  local gate="$4"
  local scheme="$5"
  local threshold="$6"
  local clip="$7"
  {
    printf '{"arm":"%s","result":' "${name}"
    uv run ternary-evaluate \
      --config configs/gated_attention_pilot.toml \
      --checkpoint "${source_checkpoint}" \
      --mode coat_a4 \
      --projection "${projection}" \
      --batches 100 \
      --qkv-quantization "${qkv}" \
      --attention-rectification "${rectification}" \
      --attention-gate "${gate}" \
      --attention-quantization "${scheme}" \
      --attention-threshold "${threshold}" \
      --attention-clip "${clip}"
    printf '}\n'
  } >> "${results}"
}

train_arm() {
  local name="$1"
  local rectification="$2"
  local gate="$3"
  local scheme="$4"
  local clip="$5"
  shift 5
  uv run ternary-train \
    --config configs/gated_attention_pilot.toml \
    --init-from "${source_checkpoint}" \
    --projection "${projection}" \
    --run-name "${name}" \
    --qkv-quantization ternary \
    --attention-rectification "${rectification}" \
    --attention-gate "${gate}" \
    --attention-quantization "${scheme}" \
    --attention-clip "${clip}" \
    "$@"
}

evaluate_arm control inherit none none prob_int2 0.125 6.0
evaluate_arm ternary-qkv ternary none none prob_int2 0.125 6.0
evaluate_arm qvit-ternary-qkv ternary qvit none prob_int2 0.125 6.0
evaluate_arm full-integer-lut ternary qvit binary score_lut_prob_int2 0.125 3.0
evaluate_arm bwta-binary-route ternary qvit binary score_lut_prob_binary 0.125 3.0

train_arm ternary-qkv none none prob_int2 6.0
train_arm qvit-gated qvit binary prob_int2 6.0
train_arm qvit-gated-distilled qvit binary score_lut_prob_int2 3.0 \
  --teacher-checkpoint "${source_checkpoint}" \
  --logit-distillation-weight 0.1 \
  --attention-distillation-weight 1.0 \
  --qk-distillation-weight 0.1 \
  --distillation-token-stride 4
