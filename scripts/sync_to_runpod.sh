#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <host> <port> <private-key>" >&2
  exit 2
fi

host="$1"
port="$2"
key="$3"
destination="/workspace/ternary-llm-experiment"
known_hosts="/tmp/ternary-llm-runpod-known-hosts"
ssh_options=(
  -i "${key}"
  -p "${port}"
  -o StrictHostKeyChecking=accept-new
  -o "UserKnownHostsFile=${known_hosts}"
)

ssh "${ssh_options[@]}" "root@${host}" "mkdir -p ${destination}"
rsync -rlptz \
  --no-owner \
  --no-group \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "data/" \
  --exclude "artifacts/" \
  --exclude ".cache/" \
  --exclude ".pytest_cache/" \
  --exclude ".ruff_cache/" \
  --exclude "__pycache__/" \
  -e "ssh ${ssh_options[*]}" \
  ./ \
  "root@${host}:${destination}/"
