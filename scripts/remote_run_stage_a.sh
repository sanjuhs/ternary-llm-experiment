#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

for mode in float ternary_weights ternary_activations ternary_forward; do
  checkpoint="artifacts/full/${mode}/checkpoint.pt"
  resume_args=()
  if [[ -f "${checkpoint}" ]]; then
    resume_args=(--resume "${checkpoint}")
  fi

  uv run ternary-train \
    --config configs/full.toml \
    --mode "${mode}" \
    --run-name "${mode}" \
    "${resume_args[@]}"

  uv run ternary-generate \
    --checkpoint "${checkpoint}" \
    --tokenizer data/full/tokenizer.json \
    --prompt "Once upon a time" \
    --max-new-tokens 160 \
    --device cuda \
    > "artifacts/full/${mode}/sample.txt"
done
