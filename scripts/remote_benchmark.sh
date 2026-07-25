#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

for mode in float ternary_forward; do
  uv run ternary-train \
    --config configs/gpu_benchmark.toml \
    --mode "${mode}" \
    --run-name "${mode}"
done
