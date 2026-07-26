import hashlib
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
