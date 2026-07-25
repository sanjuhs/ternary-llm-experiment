#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

for mode in float ternary_weights ternary_activations ternary_forward; do
  uv run ternary-train \
    --config configs/full.toml \
    --mode "${mode}" \
    --max-steps 1000 \
    --output-dir artifacts/pilot \
    --run-name "${mode}"
done
