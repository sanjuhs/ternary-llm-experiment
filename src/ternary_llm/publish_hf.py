from __future__ import annotations

import argparse
import hashlib
import json
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi

from ternary_llm.artifacts import build_run_manifest

DEFAULT_REPO_ID = "sanjuhs/ternary-llm-experiment"


def local_relative_files(folder: Path) -> tuple[str, ...]:
    """Return the regular files that upload_folder is expected to publish."""
    folder = folder.resolve()
    return tuple(
        sorted(
            path.relative_to(folder).as_posix()
            for path in folder.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(folder).parts
        )
    )


def remote_path(path_in_repo: str, relative_path: str) -> str:
    prefix = path_in_repo.strip("/")
    if not prefix:
        return relative_path
    return str(PurePosixPath(prefix, relative_path))


def missing_remote_files(
    local_files: tuple[str, ...],
    remote_files: list[str],
    *,
    path_in_repo: str,
) -> tuple[str, ...]:
    remote_set = set(remote_files)
    return tuple(
        relative_path
        for relative_path in local_files
        if remote_path(path_in_repo, relative_path) not in remote_set
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_remote_integrity(
    folder: Path,
    local_files: tuple[str, ...],
    remote_infos: list[Any],
    *,
    path_in_repo: str,
) -> dict[str, int]:
    """Verify every uploaded byte through its Git blob or LFS object digest."""
    folder = folder.resolve()
    info_by_path = {
        str(getattr(info, "path", "")): info
        for info in remote_infos
        if getattr(info, "path", None)
    }
    counts = {"git_blob_sha1": 0, "lfs_sha256": 0}
    for relative_path in local_files:
        remote_name = remote_path(path_in_repo, relative_path)
        info = info_by_path.get(remote_name)
        if info is None:
            raise RuntimeError(f"remote integrity metadata is missing: {remote_name}")
        local_path = folder / relative_path
        remote_size = getattr(info, "size", None)
        if remote_size != local_path.stat().st_size:
            raise RuntimeError(f"remote size does not match local file: {remote_name}")

        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            expected = _sha256(local_path)
            actual = getattr(lfs, "sha256", None)
            method = "lfs_sha256"
        else:
            expected = _git_blob_sha1(local_path)
            actual = getattr(info, "blob_id", None)
            method = "git_blob_sha1"
        if actual != expected:
            raise RuntimeError(
                f"remote {method} does not match local file: {remote_name}"
            )
        counts[method] += 1
    return counts


@contextmanager
def ipv4_only_dns(enabled: bool) -> Iterator[None]:
    """Temporarily omit IPv6 DNS answers for hosts with broken IPv6 routing."""
    if not enabled:
        yield
        return

    original = socket.getaddrinfo

    def getaddrinfo_ipv4(*args: Any, **kwargs: Any) -> list[Any]:
        return [
            result
            for result in original(*args, **kwargs)
            if result[0] == socket.AF_INET
        ]

    socket.getaddrinfo = getaddrinfo_ipv4
    try:
        yield
    finally:
        socket.getaddrinfo = original


def publish_run(
    run_dir: Path,
    *,
    repo_id: str,
    path_in_repo: str,
    repo_type: str,
    commit_message: str,
    ipv4_only: bool,
    api: HfApi | None = None,
) -> dict[str, Any]:
    """Checksum-audit, upload, and remotely enumerate a completed run."""
    run_dir = run_dir.resolve()
    manifest = build_run_manifest(
        run_dir,
        require_complete_checksums=True,
    )
    manifest_path = run_dir / "artifact-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = publish_verified_folder(
        run_dir,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
        commit_message=commit_message,
        ipv4_only=ipv4_only,
        api=api,
    )
    result["artifact_manifest"] = manifest
    result["publication_kind"] = "audited_run"
    return result


def publish_verified_folder(
    folder: Path,
    *,
    repo_id: str,
    path_in_repo: str,
    repo_type: str,
    commit_message: str,
    ipv4_only: bool,
    api: HfApi | None = None,
) -> dict[str, Any]:
    """Upload an arbitrary artifact folder and verify every remote byte.

    Unlike :func:`publish_run`, this does not claim the folder is a completed
    training run. It is intended for experiment selections, comparisons, and
    summaries that have their own schemas.
    """
    folder = folder.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"artifact folder does not exist: {folder}")
    expected_files = local_relative_files(folder)
    if not expected_files:
        raise FileNotFoundError(f"artifact folder contains no files: {folder}")

    hub_api = api or HfApi()
    with ipv4_only_dns(ipv4_only):
        commit = hub_api.upload_folder(
            folder_path=folder,
            repo_id=repo_id,
            path_in_repo=path_in_repo.strip("/"),
            repo_type=repo_type,
            commit_message=commit_message,
        )
        revision = getattr(commit, "oid", None)
        remote_files = hub_api.list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
        )

    missing = missing_remote_files(
        expected_files,
        remote_files,
        path_in_repo=path_in_repo,
    )
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"upload completed but remote files are missing: {missing_text}")

    expected_remote_paths = [
        remote_path(path_in_repo, relative_path)
        for relative_path in expected_files
    ]
    with ipv4_only_dns(ipv4_only):
        remote_infos = hub_api.get_paths_info(
            repo_id=repo_id,
            paths=expected_remote_paths,
            repo_type=repo_type,
            revision=revision,
        )
    remote_integrity = verify_remote_integrity(
        folder,
        expected_files,
        remote_infos,
        path_in_repo=path_in_repo,
    )

    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_in_repo": path_in_repo.strip("/"),
        "commit_oid": revision,
        "commit_url": str(commit),
        "verified_file_count": len(expected_files),
        "verified_files": list(expected_files),
        "remote_integrity": remote_integrity,
        "publication_kind": "verified_folder",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a completed run, upload it to Hugging Face, and verify filenames."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--path-in-repo")
    parser.add_argument("--repo-type", choices=("model", "dataset", "space"), default="model")
    parser.add_argument("--commit-message")
    parser.add_argument(
        "--ipv4-only",
        action="store_true",
        help="ignore IPv6 DNS results when the local network has no working IPv6 route",
    )
    parser.add_argument(
        "--metadata-folder",
        action="store_true",
        help=(
            "upload and digest-verify a non-run artifact folder without "
            "claiming checkpoint-run audit semantics"
        ),
    )
    args = parser.parse_args()

    path_in_repo = args.path_in_repo or args.run_dir.name
    commit_message = args.commit_message or f"Publish verified run {args.run_dir.name}"
    publish = publish_verified_folder if args.metadata_folder else publish_run
    result = publish(
        args.run_dir,
        repo_id=args.repo_id,
        path_in_repo=path_in_repo,
        repo_type=args.repo_type,
        commit_message=commit_message,
        ipv4_only=args.ipv4_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
