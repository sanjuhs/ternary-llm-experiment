import hashlib
import json
from pathlib import Path

import pytest

from ternary_llm.artifacts import ArtifactValidationError, build_run_manifest


def _write_completed_run(run_dir: Path) -> None:
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "resolved-config.json").write_text('{"mode": "ternary"}\n')
    (run_dir / "metrics.jsonl").write_text(
        '{"type":"train","step":1,"loss":2.0}\n'
        '{"type":"validation","step":1,"loss":1.5}\n'
    )
    (run_dir / "full-validation.json").write_text(
        '{"loss":1.5,"perplexity":4.48,"evaluated_tokens":100}\n'
    )
    (run_dir / "diagnostics.json").write_text('{"mode":"ternary"}\n')
    (run_dir / "generations.txt").write_text("Once upon a time, a fox ran.\n")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run_dir / "SHA256SUMS").write_text(f"{digest}  checkpoint.pt\n")


def test_artifact_manifest_is_deterministic_and_validates_checksum(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "strict-run"
    _write_completed_run(run_dir)

    first = build_run_manifest(run_dir)
    second = build_run_manifest(run_dir)

    assert first == second
    assert first["validation"]["loss"] == 1.5
    assert first["validation"]["logged_validation_rows"] == 1
    assert first["files"]["checkpoint.pt"]["bytes"] == len(b"checkpoint")


def test_artifact_manifest_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "strict-run"
    _write_completed_run(run_dir)
    (run_dir / "checkpoint.pt").write_bytes(b"tampered")

    with pytest.raises(ArtifactValidationError, match="does not match"):
        build_run_manifest(run_dir)


def test_artifact_manifest_validates_v2_deployment_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "strict-run"
    _write_completed_run(run_dir)
    model = run_dir / "model-2bit.pt"
    model.write_bytes(b"packed")
    (run_dir / "packed-export.json").write_text(
        json.dumps(
            {
                "source_bytes": 100,
                "packed_bytes": model.stat().st_size,
                "tensor_count": 3,
                "encoding_counts": {
                    "scaled-ternary-2bit-v1": 2,
                    "positive-scale-int16-v1": 1,
                },
                "inference_contract": {
                    "ternary_operand_contract": {"satisfied": True},
                    "end_to_end_integer_reference": {"satisfied": False},
                },
            }
        )
        + "\n"
    )

    manifest = build_run_manifest(run_dir)

    assert manifest["deployment"] == {
        "format": "ternary-deployment-v2",
        "contract_present": True,
        "tensor_count": 3,
        "encoding_counts": {
            "scaled-ternary-2bit-v1": 2,
            "positive-scale-int16-v1": 1,
        },
        "ternary_operand_contract_satisfied": True,
        "end_to_end_integer_reference_satisfied": False,
    }


def test_artifact_manifest_rejects_unpaired_packed_export(tmp_path: Path) -> None:
    run_dir = tmp_path / "strict-run"
    _write_completed_run(run_dir)
    (run_dir / "model-2bit.pt").write_bytes(b"packed")

    with pytest.raises(ArtifactValidationError, match="must be present together"):
        build_run_manifest(run_dir)


def test_artifact_manifest_rejects_bad_encoding_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "strict-run"
    _write_completed_run(run_dir)
    model = run_dir / "model-2bit.pt"
    model.write_bytes(b"packed")
    (run_dir / "packed-export.json").write_text(
        json.dumps(
            {
                "source_bytes": 100,
                "packed_bytes": model.stat().st_size,
                "tensor_count": 3,
                "encoding_counts": {"scaled-ternary-2bit-v1": 2},
                "inference_contract": {
                    "ternary_operand_contract": {"satisfied": False},
                    "end_to_end_integer_reference": {"satisfied": False},
                },
            }
        )
        + "\n"
    )

    with pytest.raises(ArtifactValidationError, match="summing to tensor_count"):
        build_run_manifest(run_dir)
