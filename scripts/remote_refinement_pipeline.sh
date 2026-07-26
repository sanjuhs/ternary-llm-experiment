#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/ternary-llm-experiment
export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

exec 9>/tmp/ternary-llm-refinement-pipeline.lock
if ! flock -n 9; then
  echo "refinement pipeline is already running" >&2
  exit 75
fi

base="artifacts/tinystories-28m"
pipeline_log="${base}/REFINEMENT_PIPELINE_STATUS"
failure_marker="${base}/REFINEMENT_PIPELINE_FAILED"
success_marker="${base}/REFINEMENT_PIPELINE_SUCCESS"

if [[ -s "${success_marker}" ]]; then
  echo "refinement pipeline already complete"
  exit 0
fi
rm -f "${failure_marker}"

current_stage="waiting-for-baseline"
record_status() {
  printf 'stage=%s\nupdated_at=%s\n' \
    "${current_stage}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${pipeline_log}"
}
on_error() {
  rc=$?
  printf 'stage=%s\nexit_code=%s\nfailed_at=%s\n' \
    "${current_stage}" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${failure_marker}"
  exit "${rc}"
}
trap on_error ERR

record_status
while pgrep -f '^bash scripts/remote_28m_baseline\.sh$' >/dev/null; do
  sleep 30
done

stages=(
  "strict-finalization:scripts/remote_finalize_strict.sh:hadamard-ternary-p3-half-pass/SUCCESS"
  "attention-clip:scripts/remote_attention_clip_refinement.sh:attention-clip-refinement/SUCCESS"
  "shared-qkv-scale:scripts/remote_shared_qkv_scale_refinement.sh:shared-qkv-scale-refinement/SUCCESS"
  "softmax1:scripts/remote_softmax1_refinement.sh:softmax1-refinement/SUCCESS"
  "relu-hardening:scripts/remote_relu_hardening.sh:relu-hardening/SUCCESS"
  "binary-qk:scripts/remote_binary_qk_fallback.sh:binary-qk-fallback/SUCCESS"
  "integer-rmsnorm:scripts/remote_integer_rmsnorm_screen.sh:integer-rmsnorm-screen/SUCCESS"
  "strict-contract:scripts/remote_strict_contract_endpoint.sh:strict-contract-refinement/SUCCESS"
)

for declaration in "${stages[@]}"; do
  IFS=: read -r current_stage script marker <<< "${declaration}"
  record_status
  bash "${script}"
  if [[ ! -s "${base}/${marker}" ]]; then
    echo "${current_stage} returned without a SUCCESS marker" >&2
    false
  fi
done

current_stage="overnight-summary"
record_status
/opt/ternary-llm-venv/bin/ternary-overnight-summary \
  "${base}" \
  --json-output "${base}/overnight-summary/summary.json" \
  --markdown-output "${base}/overnight-summary/summary.md"

current_stage="complete"
record_status
printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${success_marker}"
rm -f "${failure_marker}"
