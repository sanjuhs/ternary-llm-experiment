#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv

for lanes in 2 4; do
  uv run ternary-train \
    --config configs/full.toml \
    --mode population_ternary \
    --population-lanes "${lanes}" \
    --max-steps 1000 \
    --output-dir artifacts/population-pilot \
    --run-name "p${lanes}"
done
