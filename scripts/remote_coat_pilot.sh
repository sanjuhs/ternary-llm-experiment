#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

source_checkpoint="artifacts/full-stage-a/ternary_weights/checkpoint.pt"
projection="artifacts/coat/ternary-weights-projection.pt"

for mode in hadamard_a4 coat_a4 hadamard_ternary coat_ternary; do
  arguments=(
    --config configs/coat_pilot.toml
    --mode "${mode}"
    --init-from "${source_checkpoint}"
    --run-name "${mode}"
  )
  if [[ "${mode}" == coat_* ]]; then
    arguments+=(--projection "${projection}")
  fi
  uv run ternary-train "${arguments[@]}"
done
