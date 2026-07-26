from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_REQUIRED_FILES = (
    "checkpoint.pt",
    "resolved-config.json",
    "metrics.jsonl",
    "full-validation.json",
    "diagnostics.json",
    "generations.txt",
)


class ArtifactValidationError(ValueError):
    """Raised when a completed experiment run is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{path.name} must contain a JSON object")
    return payload


def _validate_full_validation(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    for name in ("loss", "perplexity"):
        value = payload.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ArtifactValidationError(
                f"{path.name} needs a finite positive {name}"
            )
    evaluated_tokens = payload.get("evaluated_tokens")
    if (
        not isinstance(evaluated_tokens, (int, float))
        or not math.isfinite(evaluated_tokens)
        or evaluated_tokens <= 0
        or not float(evaluated_tokens).is_integer()
    ):
        raise ArtifactValidationError(
            f"{path.name} needs a positive integer-valued evaluated_tokens count"
        )
    payload["evaluated_tokens"] = int(evaluated_tokens)
    return payload


def _validate_metrics(path: Path) -> int:
    validation_rows = 0
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ArtifactValidationError(f"cannot read {path.name}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArtifactValidationError(
                f"{path.name}:{line_number} is not valid JSON"
            ) from error
        if not isinstance(row, dict):
            raise ArtifactValidationError(
                f"{path.name}:{line_number} must contain an object"
            )
        validation_rows += row.get("type") == "validation"
    if validation_rows == 0:
        raise ArtifactValidationError(f"{path.name} contains no validation rows")
    return validation_rows


def _verify_declared_checkpoint_hash(run_dir: Path, checkpoint_hash: str) -> None:
    checksum_path = run_dir / "SHA256SUMS"
    if not checksum_path.exists():
        return
    declarations = {}
    for line in checksum_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            declarations[Path(parts[-1].lstrip("*")).name] = parts[0]
    declared = declarations.get("checkpoint.pt")
    if declared is None:
        raise ArtifactValidationError("SHA256SUMS does not declare checkpoint.pt")
    if declared != checkpoint_hash:
        raise ArtifactValidationError(
            "SHA256SUMS checkpoint hash does not match checkpoint.pt"
        )


def build_run_manifest(
    run_dir: Path,
    *,
    required_files: tuple[str, ...] = DEFAULT_REQUIRED_FILES,
) -> dict[str, Any]:
    """Validate a completed run and return a deterministic content manifest."""
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise ArtifactValidationError(f"run directory does not exist: {run_dir}")
    files = {}
    for name in required_files:
        path = run_dir / name
        if not path.is_file():
            raise ArtifactValidationError(f"required artifact is missing: {name}")
        size = path.stat().st_size
        if size <= 0:
            raise ArtifactValidationError(f"required artifact is empty: {name}")
        files[name] = {"bytes": size, "sha256": _sha256(path)}

    _load_json_object(run_dir / "resolved-config.json")
    _load_json_object(run_dir / "diagnostics.json")
    full_validation = _validate_full_validation(run_dir / "full-validation.json")
    validation_rows = _validate_metrics(run_dir / "metrics.jsonl")
    _verify_declared_checkpoint_hash(
        run_dir,
        files["checkpoint.pt"]["sha256"],
    )

    optional_files = ("SHA256SUMS", "model-2bit.pt", "packed-export.json")
    for name in optional_files:
        path = run_dir / name
        if path.is_file() and path.stat().st_size > 0:
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}

    return {
        "run": run_dir.name,
        "validation": {
            "loss": full_validation["loss"],
            "perplexity": full_validation["perplexity"],
            "evaluated_tokens": full_validation["evaluated_tokens"],
            "logged_validation_rows": validation_rows,
        },
        "files": dict(sorted(files.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--require-file",
        action="append",
        default=[],
        help="additional required filename relative to every run directory",
    )
    args = parser.parse_args()
    required = (*DEFAULT_REQUIRED_FILES, *args.require_file)
    manifests = [
        build_run_manifest(run_dir, required_files=required)
        for run_dir in args.run_dirs
    ]
    print(json.dumps({"runs": manifests}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
