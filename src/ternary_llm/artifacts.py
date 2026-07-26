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


def _checksum_declarations(run_dir: Path) -> dict[str, str]:
    checksum_path = run_dir / "SHA256SUMS"
    if not checksum_path.exists():
        return {}
    declarations: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text().splitlines(),
        start=1,
    ):
        parts = line.split()
        if len(parts) < 2:
            raise ArtifactValidationError(
                f"SHA256SUMS:{line_number} is not a checksum declaration"
            )
        digest = parts[0].lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ArtifactValidationError(
                f"SHA256SUMS:{line_number} has an invalid SHA-256 digest"
            )
        name = Path(parts[-1].lstrip("*")).name
        existing = declarations.get(name)
        if existing is not None and existing != digest:
            raise ArtifactValidationError(
                f"SHA256SUMS declares conflicting hashes for {name}"
            )
        declarations[name] = digest
    return declarations


def _verify_declared_hashes(
    run_dir: Path,
    files: dict[str, dict[str, Any]],
    *,
    require_complete: bool,
) -> None:
    declarations = _checksum_declarations(run_dir)
    if not declarations:
        if require_complete:
            raise ArtifactValidationError("complete SHA256SUMS is required")
        return
    if "checkpoint.pt" not in declarations:
        raise ArtifactValidationError("SHA256SUMS does not declare checkpoint.pt")

    expected = set(files) - {"SHA256SUMS"}
    if require_complete:
        missing = sorted(expected - declarations.keys())
        if missing:
            raise ArtifactValidationError(
                "SHA256SUMS is missing required artifacts: " + ", ".join(missing)
            )

    for name, declared in declarations.items():
        path = run_dir / name
        if not path.is_file():
            raise ArtifactValidationError(
                f"SHA256SUMS declares missing artifact: {name}"
            )
        actual = _sha256(path)
        if declared != actual:
            raise ArtifactValidationError(
                f"SHA256SUMS hash for {name} does not match the file"
            )


def _validate_packed_export(run_dir: Path) -> dict[str, Any] | None:
    model_path = run_dir / "model-2bit.pt"
    metadata_path = run_dir / "packed-export.json"
    if not model_path.exists() and not metadata_path.exists():
        return None
    if not model_path.is_file() or not metadata_path.is_file():
        raise ArtifactValidationError(
            "model-2bit.pt and packed-export.json must be present together"
        )
    if model_path.stat().st_size <= 0:
        raise ArtifactValidationError("model-2bit.pt is empty")

    payload = _load_json_object(metadata_path)
    for name in ("source_bytes", "packed_bytes", "tensor_count"):
        value = payload.get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or not float(value).is_integer()
        ):
            raise ArtifactValidationError(
                f"packed-export.json needs a positive integer-valued {name}"
            )
    if int(payload["packed_bytes"]) != model_path.stat().st_size:
        raise ArtifactValidationError(
            "packed-export.json packed_bytes does not match model-2bit.pt"
        )

    contract = payload.get("inference_contract")
    if contract is None:
        return {
            "format": "legacy-ternary-2bit-v1",
            "contract_present": False,
            "tensor_count": int(payload["tensor_count"]),
        }
    if not isinstance(contract, dict):
        raise ArtifactValidationError("inference_contract must be an object")
    operand_contract = contract.get("ternary_operand_contract")
    integer_reference = contract.get("end_to_end_integer_reference")
    if not isinstance(operand_contract, dict) or not isinstance(
        operand_contract.get("satisfied"),
        bool,
    ):
        raise ArtifactValidationError(
            "inference_contract needs a Boolean ternary_operand_contract.satisfied"
        )
    if not isinstance(integer_reference, dict) or not isinstance(
        integer_reference.get("satisfied"),
        bool,
    ):
        raise ArtifactValidationError(
            "inference_contract needs a Boolean end_to_end_integer_reference.satisfied"
        )
    encoding_counts = payload.get("encoding_counts")
    if (
        not isinstance(encoding_counts, dict)
        or not encoding_counts
        or any(
            not isinstance(count, int) or count <= 0
            for count in encoding_counts.values()
        )
        or sum(encoding_counts.values()) != int(payload["tensor_count"])
    ):
        raise ArtifactValidationError(
            "encoding_counts must be positive integers summing to tensor_count"
        )
    return {
        "format": "ternary-deployment-v2",
        "contract_present": True,
        "tensor_count": int(payload["tensor_count"]),
        "encoding_counts": encoding_counts,
        "ternary_operand_contract_satisfied": operand_contract["satisfied"],
        "end_to_end_integer_reference_satisfied": integer_reference["satisfied"],
    }


def build_run_manifest(
    run_dir: Path,
    *,
    required_files: tuple[str, ...] = DEFAULT_REQUIRED_FILES,
    require_complete_checksums: bool = False,
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
    deployment = _validate_packed_export(run_dir)

    optional_files = {
        "SHA256SUMS",
        "model-2bit.pt",
        "packed-export.json",
    }
    if (run_dir / "best-checkpoint.pt").is_file():
        optional_files.add("best-checkpoint.pt")
    optional_files.update(
        path.name
        for path in run_dir.glob("checkpoint-step-*.pt")
        if path.is_file()
    )
    for name in sorted(optional_files):
        path = run_dir / name
        if path.is_file() and path.stat().st_size > 0:
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}

    _verify_declared_hashes(
        run_dir,
        files,
        require_complete=require_complete_checksums,
    )

    manifest = {
        "run": run_dir.name,
        "validation": {
            "loss": full_validation["loss"],
            "perplexity": full_validation["perplexity"],
            "evaluated_tokens": full_validation["evaluated_tokens"],
            "logged_validation_rows": validation_rows,
        },
        "files": dict(sorted(files.items())),
    }
    if deployment is not None:
        manifest["deployment"] = deployment
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--require-file",
        action="append",
        default=[],
        help="additional required filename relative to every run directory",
    )
    parser.add_argument(
        "--require-complete-checksums",
        action="store_true",
        help=(
            "require SHA256SUMS to declare every required and deployment "
            "artifact in the run"
        ),
    )
    args = parser.parse_args()
    required = (*DEFAULT_REQUIRED_FILES, *args.require_file)
    manifests = [
        build_run_manifest(
            run_dir,
            required_files=required,
            require_complete_checksums=args.require_complete_checksums,
        )
        for run_dir in args.run_dirs
    ]
    print(json.dumps({"runs": manifests}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
