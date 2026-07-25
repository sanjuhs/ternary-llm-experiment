#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

config="configs/tinystories_28m.toml"
output="artifacts/tinystories-28m"

uv run ternary-train \
  --config "${config}" \
  --run-name float-two-pass

float_checkpoint="${output}/float-two-pass/checkpoint.pt"

uv run ternary-train \
  --config "${config}" \
  --mode ternary_weights \
  --init-from "${float_checkpoint}" \
  --teacher-checkpoint "${float_checkpoint}" \
  --run-name ternary-weights-one-pass \
  --max-steps 59592 \
  --logit-distillation-weight 0.1 \
  --hidden-distillation-weight 0.5

ternary_checkpoint="${output}/ternary-weights-one-pass/checkpoint.pt"
projection="${output}/ternary-weights-projection.pt"

uv run ternary-calibrate-projection \
  --checkpoint "${ternary_checkpoint}" \
  --config "${config}" \
  --output "${projection}" \
  --device cuda \
  --batches 24 \
  --batch-size 4

uv run ternary-train \
  --config "${config}" \
  --mode coat_a4 \
  --init-from "${ternary_checkpoint}" \
  --projection "${projection}" \
  --teacher-checkpoint "${float_checkpoint}" \
  --run-name coat-a4-quarter-pass \
  --max-steps 15000 \
  --logit-distillation-weight 0.1 \
  --hidden-distillation-weight 0.5

coat_checkpoint="${output}/coat-a4-quarter-pass/checkpoint.pt"

uv run ternary-train \
  --config "${config}" \
  --mode hadamard_progressive \
  --init-from "${coat_checkpoint}" \
  --teacher-checkpoint "${float_checkpoint}" \
  --run-name hadamard-ternary-p3-half-pass \
  --activation-encoding residual_ternary \
  --activation-planes 3 \
  --qkv-quantization ternary \
  --attention-quantization score_lut_prob_int2 \
  --attention-clip 3.0 \
  --max-steps 30000 \
  --logit-distillation-weight 0.1 \
  --attention-distillation-weight 0.5 \
  --hidden-distillation-weight 1.0
