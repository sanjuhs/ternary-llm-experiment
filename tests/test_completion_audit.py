import json
from pathlib import Path
from typing import Any

import pytest

from ternary_llm.completion_audit import (
    METADATA_FOLDERS,
    CompletionAuditError,
    build_completion_audit,
)
from ternary_llm.publish_hf import DEFAULT_REPO_ID, local_relative_files


def _receipt(
    folder: Path,
    *,
    kind: str,
    path_in_repo: str,
) -> dict[str, Any]:
    files = list(local_relative_files(folder))
    return {
        "repo_id": DEFAULT_REPO_ID,
        "publication_kind": kind,
        "commit_oid": "commit-123",
        "path_in_repo": path_in_repo,
        "verified_files": files,
        "verified_file_count": len(files),
        "remote_integrity": {
            "git_blob_sha1": len(files),
            "lfs_sha256": 0,
        },
    }


def _fixture(root: Path, receipts: Path) -> None:
    run = root / "run-a"
    run.mkdir(parents=True)
    (run / "checkpoint.pt").write_bytes(b"run")
    receipts.mkdir()
    (receipts / "run--run-a.json").write_text(
        json.dumps(
            _receipt(
                run,
                kind="audited_run",
                path_in_repo="tinystories-28m/runs/run-a",
            )
        )
    )
    for name in METADATA_FOLDERS:
        folder = root / name
        folder.mkdir()
        (folder / "result.json").write_text("{}\n")
        (receipts / f"metadata--{name}.json").write_text(
            json.dumps(
                _receipt(
                    folder,
                    kind="verified_folder",
                    path_in_repo=f"tinystories-28m/experiments/{name}",
                )
            )
        )


def _summary(_root: Path) -> dict[str, Any]:
    return {
        "best_run": {"run": "run-a", "loss": 2.0},
        "audited_runs": [{"run": "run-a"}],
    }


def test_completion_audit_binds_current_files_to_receipts(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    receipts = tmp_path / "receipts"
    _fixture(root, receipts)

    audit = build_completion_audit(
        root,
        receipts,
        summary_builder=_summary,
    )

    assert audit["status"] == "complete"
    assert audit["verified_publication_count"] == 1 + len(METADATA_FOLDERS)


def test_completion_audit_rejects_local_change_after_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    receipts = tmp_path / "receipts"
    _fixture(root, receipts)
    (root / "run-a" / "new-file.txt").write_text("changed\n")

    with pytest.raises(CompletionAuditError, match="changed after"):
        build_completion_audit(
            root,
            receipts,
            summary_builder=_summary,
        )


def test_completion_audit_rejects_missing_receipt(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    receipts = tmp_path / "receipts"
    _fixture(root, receipts)
    (receipts / "metadata--relu-hardening.json").unlink()

    with pytest.raises(CompletionAuditError, match="receipt is missing"):
        build_completion_audit(
            root,
            receipts,
            summary_builder=_summary,
        )
