#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv

uv run ternary-train \
  --config configs/full.toml \
  --mode ternary_forward \
  --residual-scale 0.25 \
  --max-steps 1000 \
  --output-dir artifacts/residual-scale-pilot \
  --run-name strict-r025

uv run ternary-train \
  --config configs/full.toml \
  --mode population_ternary \
  --population-lanes 4 \
  --residual-scale 0.25 \
  --max-steps 1000 \
  --output-dir artifacts/residual-scale-pilot \
  --run-name p4-r025
