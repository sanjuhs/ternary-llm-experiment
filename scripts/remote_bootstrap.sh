#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ternary-llm-experiment

export UV_PROJECT_ENVIRONMENT=/opt/ternary-llm-venv
export UV_LINK_MODE=copy

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv sync --extra dev
uv run ruff check .
uv run pytest

if [[ ! -f data/full/metadata.json ]]; then
  uv run ternary-data download
  uv run ternary-data prepare \
    --output-dir data/processed \
    --max-train-stories 50000 \
    --max-validation-stories 2000
  uv run ternary-data prepare \
    --full \
    --output-dir data/full \
    --tokenizer-from data/processed/tokenizer.json
else
  echo "Using existing full-corpus token stream."
fi

nvidia-smi
uv run python - <<'PY'
import json
from pathlib import Path

import torch

metadata = json.loads(Path("data/full/metadata.json").read_text())
print(
    json.dumps(
        {
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "tokens": metadata["tokens"],
        },
        indent=2,
    )
)
PY
