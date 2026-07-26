from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ternary_llm.artifacts import ArtifactValidationError
from ternary_llm.publish_hf import (
    ipv4_only_dns,
    local_relative_files,
    missing_remote_files,
    publish_run,
    remote_path,
)


def test_local_relative_files_are_sorted_and_ignore_git(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "z.txt").write_text("z")
    (tmp_path / "nested" / "a.txt").write_text("a")
    (tmp_path / ".git" / "config").write_text("private")

    assert local_relative_files(tmp_path) == ("nested/a.txt", "z.txt")


def test_remote_path_normalizes_prefix() -> None:
    assert remote_path("/runs/example/", "metrics.jsonl") == "runs/example/metrics.jsonl"
    assert remote_path("", "metrics.jsonl") == "metrics.jsonl"


def test_missing_remote_files_uses_repo_prefix() -> None:
    missing = missing_remote_files(
        ("checkpoint.pt", "metrics.jsonl"),
        ["runs/demo/checkpoint.pt", "README.md"],
        path_in_repo="runs/demo",
    )

    assert missing == ("metrics.jsonl",)


def test_ipv4_context_does_not_patch_when_disabled() -> None:
    import socket

    original = socket.getaddrinfo
    with ipv4_only_dns(False):
        assert socket.getaddrinfo is original
    assert socket.getaddrinfo is original


def test_publish_run_rejects_incomplete_checksums_before_upload(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "resolved-config.json").write_text("{}\n")
    (run_dir / "metrics.jsonl").write_text(
        '{"type":"validation","loss":1.5}\n'
    )
    (run_dir / "full-validation.json").write_text(
        '{"loss":1.5,"perplexity":4.48,"evaluated_tokens":100}\n'
    )
    (run_dir / "diagnostics.json").write_text("{}\n")
    (run_dir / "generations.txt").write_text("Once upon a time.\n")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run_dir / "SHA256SUMS").write_text(f"{digest}  checkpoint.pt\n")

    class UnexpectedApi:
        def upload_folder(self, **kwargs: object) -> None:
            raise AssertionError(f"upload should not be called: {kwargs}")

    with pytest.raises(
        ArtifactValidationError,
        match="missing required artifacts",
    ):
        publish_run(
            run_dir,
            repo_id="owner/repo",
            path_in_repo="run",
            repo_type="model",
            commit_message="test",
            ipv4_only=False,
            api=UnexpectedApi(),  # type: ignore[arg-type]
        )
