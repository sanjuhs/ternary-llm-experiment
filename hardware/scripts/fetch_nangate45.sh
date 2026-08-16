#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output="$root_dir/hardware/lib/NangateOpenCellLibrary_typical.lib"
expected="8d540a4d4cf6d09d27c87ad067857a9c0c2eeb023ab7a56e058cd3113db4e9b1"
url="https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib"

mkdir -p "$(dirname "$output")"
curl -L --fail --max-time 120 -o "$output" "$url"
actual="$(shasum -a 256 "$output" | awk '{print $1}')"
if [[ "$actual" != "$expected" ]]; then
  echo "Nangate45 checksum mismatch: expected $expected, got $actual" >&2
  exit 1
fi
echo "$output"
