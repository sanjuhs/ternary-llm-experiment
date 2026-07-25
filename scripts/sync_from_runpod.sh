#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <host> <port> <private-key>" >&2
  exit 2
fi

host="$1"
port="$2"
key="$3"
source="/workspace/ternary-llm-experiment/artifacts/"
known_hosts="/tmp/ternary-llm-runpod-known-hosts"
ssh_options=(
  -i "${key}"
  -p "${port}"
  -o StrictHostKeyChecking=accept-new
  -o "UserKnownHostsFile=${known_hosts}"
)

mkdir -p artifacts
rsync -rlptz \
  --no-owner \
  --no-group \
  --partial \
  -e "ssh ${ssh_options[*]}" \
  "root@${host}:${source}" \
  artifacts/
