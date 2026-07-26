#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

run_dir="artifacts/tinystories-28m/hadamard-ternary-p3-half-pass"

if [[ -s "${run_dir}/SUCCESS" ]]; then
  echo "strict run is already finalized: ${run_dir}/SUCCESS"
  exit 0
fi

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

preserved_checkpoint="${run_dir}/checkpoint-step-10000.pt"
preserved_run="artifacts/tinystories-28m/hadamard-ternary-p3-step-10000-preserved"
if [[ -s "${preserved_checkpoint}" && ! -s "${preserved_run}/SUCCESS" ]]; then
  mkdir -p "${preserved_run}"
  cp --reflink=auto "${preserved_checkpoint}" "${preserved_run}/checkpoint.pt"
  cp "${run_dir}/resolved-config.json" "${preserved_run}/resolved-config.json"
  /opt/ternary-llm-venv/bin/python - \
    "${run_dir}/metrics.jsonl" \
    "${preserved_run}/metrics.jsonl" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
rows = [
    line
    for line in source.read_text().splitlines()
    if line and json.loads(line).get("step", 0) <= 10_000
]
target.write_text("\n".join(rows) + "\n")
PY

  uv run ternary-evaluate \
    --config configs/tinystories_28m.toml \
    --checkpoint "${preserved_run}/checkpoint.pt" \
    --device cuda \
    --sequential \
    > "${preserved_run}/full-validation.json"

  uv run ternary-diagnostics \
    --checkpoint "${preserved_run}/checkpoint.pt" \
    --config configs/tinystories_28m.toml \
    --device cuda \
    > "${preserved_run}/diagnostics.json"

  : > "${preserved_run}/generations.txt"
  for prompt in \
    "Once upon a time" \
    "Lily found a tiny red door" \
    "Tom wanted to help his friend"
  do
    uv run ternary-generate \
      --checkpoint "${preserved_run}/checkpoint.pt" \
      --tokenizer data/full/tokenizer.json \
      --prompt "${prompt}" \
      --max-new-tokens 120 \
      --temperature 0.8 \
      --top-k 50 \
      --device cuda \
      >> "${preserved_run}/generations.txt"
  done

  uv run ternary-export \
    --checkpoint "${preserved_run}/checkpoint.pt" \
    --output "${preserved_run}/model-2bit.pt" \
    > "${preserved_run}/packed-export.json"

  (
    cd "${preserved_run}"
    sha256sum \
      checkpoint.pt \
      resolved-config.json \
      metrics.jsonl \
      full-validation.json \
      diagnostics.json \
      generations.txt \
      model-2bit.pt \
      packed-export.json \
      > SHA256SUMS
  )
  uv run ternary-audit-artifacts \
    --require-complete-checksums \
    "${preserved_run}" \
    > "${preserved_run}/artifact-audit.json"
  printf 'source_step=10000\nrun_dir=%s\n' \
    "${preserved_run}" > "${preserved_run}/SUCCESS"
fi

uv run ternary-export \
  --checkpoint "${run_dir}/checkpoint.pt" \
  --output "${run_dir}/model-2bit.pt" \
  > "${run_dir}/packed-export.json"

(
  cd "${run_dir}"
  checksum_files=(
    checkpoint.pt \
    resolved-config.json \
    metrics.jsonl \
    full-validation.json \
    diagnostics.json \
    generations.txt \
    model-2bit.pt \
    packed-export.json
  )
  if [[ -f best-checkpoint.pt ]]; then
    checksum_files+=(best-checkpoint.pt)
  fi
  for preserved_checkpoint in checkpoint-step-*.pt; do
    if [[ -f "${preserved_checkpoint}" ]]; then
      checksum_files+=("${preserved_checkpoint}")
    fi
  done
  sha256sum "${checksum_files[@]}" > SHA256SUMS
)

uv run ternary-audit-artifacts \
  --require-complete-checksums \
  "${run_dir}" \
  > "${run_dir}/artifact-audit.json"

printf 'run_dir=%s\n' "${run_dir}" > "${run_dir}/SUCCESS"
