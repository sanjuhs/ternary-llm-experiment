#!/usr/bin/env bash

# Shared retry guards for the RunPod refinement scripts. The caller is expected
# to use `set -euo pipefail` and to have selected the project environment.

checkpoint_has_step() {
  local run_dir="$1"
  local expected_step="$2"
  local checkpoint="${run_dir}/checkpoint.pt"
  [[ -s "${checkpoint}" ]] || return 1
  /opt/ternary-llm-venv/bin/python - "${checkpoint}" "${expected_step}" <<'PY'
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(0 if int(checkpoint["step"]) == int(sys.argv[2]) else 1)
PY
}

prepare_training_run() {
  local run_dir="$1"
  local expected_step="$2"
  if checkpoint_has_step "${run_dir}" "${expected_step}"; then
    echo "reusing completed training checkpoint: ${run_dir}/checkpoint.pt"
    return 1
  fi
  if [[ -e "${run_dir}" ]]; then
    local suffix
    suffix="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    local archived="${run_dir}.incomplete-${suffix}"
    mv "${run_dir}" "${archived}"
    echo "preserved incomplete run as ${archived}" >&2
  fi
  return 0
}
