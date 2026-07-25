#!/usr/bin/env bash
set -euo pipefail

for mode in float ternary_weights ternary_activations ternary_forward; do
  uv run ternary-train \
    --config configs/tiny.toml \
    --mode "${mode}" \
    --run-name "${mode}"
done
