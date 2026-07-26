#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

run_dir="artifacts/tinystories-28m/hadamard-ternary-p3-half-pass"

for required in \
  checkpoint.pt \
  resolved-config.json \
  metrics.jsonl \
  full-validation.json \
  diagnostics.json \
  generations.txt
do
  if [[ ! -s "${run_dir}/${required}" ]]; then
    echo "strict run is not ready: missing ${required}" >&2
    exit 1
  fi
done

uv run ternary-export \
  --checkpoint "${run_dir}/checkpoint.pt" \
  --output "${run_dir}/model-2bit.pt" \
  > "${run_dir}/packed-export.json"

sha256sum "${run_dir}/checkpoint.pt" > "${run_dir}/SHA256SUMS"

uv run ternary-audit-artifacts "${run_dir}" \
  > "${run_dir}/artifact-audit.json"

printf 'run_dir=%s\n' "${run_dir}" > "${run_dir}/SUCCESS"
