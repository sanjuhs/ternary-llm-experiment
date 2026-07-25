#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

uv run ternary-train \
  --config configs/residual_refinement_pilot.toml \
  --mode hadamard_progressive \
  --init-from \
    artifacts/residual-refinement-pilot/hadamard-ternary-p3-long/checkpoint.pt \
  --teacher-checkpoint artifacts/attention-pilot/prob-int2/checkpoint.pt \
  --run-name hadamard-late-p3-mixed \
  --activation-encoding residual_ternary \
  --activation-planes 2 \
  --activation-planes-by-layer 2 2 2 3 3 3 \
  --max-steps 2500

uv run ternary-train \
  --config configs/residual_refinement_pilot.toml \
  --mode hadamard_progressive \
  --init-from \
    artifacts/residual-refinement-pilot/hadamard-ternary-p3-long/checkpoint.pt \
  --teacher-checkpoint artifacts/attention-pilot/prob-int2/checkpoint.pt \
  --run-name hadamard-ternary-p3-relu-harden \
  --activation-encoding residual_ternary \
  --activation-planes 3 \
  --feed-forward-activation relu \
  --max-steps 1500

for run_name in \
  hadamard-binary-p2-long \
  hadamard-ternary-p3-long \
  hadamard-late-p3-mixed \
  hadamard-ternary-p3-relu-harden
do
  run_dir="artifacts/residual-refinement-pilot/${run_name}"
  uv run ternary-evaluate \
    --config configs/residual_refinement_pilot.toml \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --device cuda \
    --sequential \
    > "${run_dir}/full-validation.json"
  uv run ternary-diagnostics \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --config configs/residual_refinement_pilot.toml \
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
done

config="configs/tinystories_28m.toml"
output="artifacts/tinystories-28m"

record_checkpoint() {
  local run_name=$1
  local checkpoint="${output}/${run_name}/checkpoint.pt"
  local run_dir="${output}/${run_name}"

  uv run ternary-evaluate \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --device cuda \
    --sequential \
    > "${run_dir}/full-validation.json"

  uv run ternary-diagnostics \
    --checkpoint "${checkpoint}" \
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
      --checkpoint "${checkpoint}" \
      --tokenizer data/full/tokenizer.json \
      --prompt "${prompt}" \
      --max-new-tokens 120 \
      --temperature 0.8 \
      --top-k 50 \
      --device cuda \
      >> "${run_dir}/generations.txt"
  done
}

uv run ternary-train \
  --config "${config}" \
  --run-name float-two-pass

float_checkpoint="${output}/float-two-pass/checkpoint.pt"
record_checkpoint float-two-pass

uv run ternary-train \
  --config "${config}" \
  --mode ternary_weights \
  --init-from "${float_checkpoint}" \
  --teacher-checkpoint "${float_checkpoint}" \
  --run-name ternary-weights-one-pass \
  --max-steps 59592 \
  --learning-rate 0.00003 \
  --min-learning-rate 0.000003 \
  --warmup-steps 500 \
  --weight-decay 0 \
  --logit-distillation-weight 0.1 \
  --hidden-distillation-weight 0.5

ternary_checkpoint="${output}/ternary-weights-one-pass/checkpoint.pt"
projection="${output}/ternary-weights-projection.pt"
record_checkpoint ternary-weights-one-pass
uv run ternary-export \
  --checkpoint "${ternary_checkpoint}" \
  --output "${output}/ternary-weights-one-pass/model-2bit.pt" \
  > "${output}/ternary-weights-one-pass/packed-export.json"

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
  --learning-rate 0.00003 \
  --min-learning-rate 0.000003 \
  --warmup-steps 200 \
  --weight-decay 0 \
  --logit-distillation-weight 0.1 \
  --hidden-distillation-weight 0.5

coat_checkpoint="${output}/coat-a4-quarter-pass/checkpoint.pt"
record_checkpoint coat-a4-quarter-pass

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
  --learning-rate 0.00003 \
  --min-learning-rate 0.000003 \
  --warmup-steps 500 \
  --weight-decay 0 \
  --logit-distillation-weight 0.1 \
  --attention-distillation-weight 0.5 \
  --hidden-distillation-weight 1.0

record_checkpoint hadamard-ternary-p3-half-pass
