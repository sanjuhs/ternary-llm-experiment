from __future__ import annotations

from pathlib import Path

from ternary_llm.publish_hf import (
    ipv4_only_dns,
    local_relative_files,
    missing_remote_files,
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
