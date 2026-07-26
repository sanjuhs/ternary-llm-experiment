from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ternary_llm.artifacts import ArtifactValidationError
from ternary_llm.overnight_summary import (
    OvernightSummaryError,
    build_overnight_summary,
)
from ternary_llm.publish_hf import DEFAULT_REPO_ID, local_relative_files

METADATA_FOLDERS = (
    "attention-clip-refinement",
    "shared-qkv-scale-refinement",
    "softmax1-refinement",
    "relu-hardening",
    "binary-qk-fallback",
    "integer-rmsnorm-screen",
    "overnight-summary",
)


class CompletionAuditError(ValueError):
    """Raised when final local or remote-preservation evidence is incomplete."""


SummaryBuilder = Callable[[Path], dict[str, Any]]


def _load_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CompletionAuditError(f"publication receipt is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CompletionAuditError(f"publication receipt is invalid: {path}") from error
    if not isinstance(payload, dict):
        raise CompletionAuditError(f"publication receipt must be an object: {path}")
    return payload


def _verify_receipt(
    receipt_path: Path,
    folder: Path,
    *,
    repo_id: str,
    publication_kind: str,
) -> dict[str, Any]:
    if not folder.is_dir():
        raise CompletionAuditError(f"published local folder is missing: {folder}")
    receipt = _load_receipt(receipt_path)
    if receipt.get("repo_id") != repo_id:
        raise CompletionAuditError(f"receipt repo mismatch: {receipt_path}")
    if receipt.get("publication_kind") != publication_kind:
        raise CompletionAuditError(f"receipt publication kind mismatch: {receipt_path}")
    if not isinstance(receipt.get("commit_oid"), str) or not receipt["commit_oid"]:
        raise CompletionAuditError(f"receipt has no commit OID: {receipt_path}")

    current_files = local_relative_files(folder)
    verified_files = receipt.get("verified_files")
    if verified_files != list(current_files):
        raise CompletionAuditError(
            f"local files changed after verified publication: {folder}"
        )
    count = receipt.get("verified_file_count")
    if count != len(current_files) or count <= 0:
        raise CompletionAuditError(f"receipt file count mismatch: {receipt_path}")
    integrity = receipt.get("remote_integrity")
    digest_counts = (
        integrity.get("git_blob_sha1"),
        integrity.get("lfs_sha256"),
    ) if isinstance(integrity, dict) else ()
    if (
        len(digest_counts) != 2
        or any(not isinstance(value, int) or value < 0 for value in digest_counts)
        or sum(digest_counts) != count
    ):
        raise CompletionAuditError(f"receipt digest count mismatch: {receipt_path}")
    return {
        "folder": folder.name,
        "commit_oid": receipt["commit_oid"],
        "path_in_repo": receipt.get("path_in_repo"),
        "verified_file_count": count,
        "remote_integrity": integrity,
    }


def build_completion_audit(
    artifacts_root: Path,
    receipts_dir: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    summary_builder: SummaryBuilder = build_overnight_summary,
) -> dict[str, Any]:
    """Prove local completion and bind it to verified Hub publication receipts."""
    artifacts_root = artifacts_root.resolve()
    receipts_dir = receipts_dir.resolve()
    try:
        summary = summary_builder(artifacts_root)
    except (OvernightSummaryError, ArtifactValidationError) as error:
        raise CompletionAuditError(
            f"experiment summary is incomplete: {error}"
        ) from error

    run_publications = []
    for manifest in summary["audited_runs"]:
        run_name = str(manifest["run"])
        run_publications.append(
            _verify_receipt(
                receipts_dir / f"run--{run_name}.json",
                artifacts_root / run_name,
                repo_id=repo_id,
                publication_kind="audited_run",
            )
        )

    metadata_publications = [
        _verify_receipt(
            receipts_dir / f"metadata--{name}.json",
            artifacts_root / name,
            repo_id=repo_id,
            publication_kind="verified_folder",
        )
        for name in METADATA_FOLDERS
    ]
    paths = [
        publication["path_in_repo"]
        for publication in (*run_publications, *metadata_publications)
    ]
    if any(not isinstance(path, str) or not path for path in paths):
        raise CompletionAuditError("every publication needs a non-empty Hub path")
    if len(paths) != len(set(paths)):
        raise CompletionAuditError("publication receipts reuse a Hub path")

    return {
        "status": "complete",
        "repo_id": repo_id,
        "best_run": summary["best_run"],
        "run_publications": run_publications,
        "metadata_publications": metadata_publications,
        "verified_publication_count": len(paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts_root", type=Path)
    parser.add_argument("receipts_dir", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    audit = build_completion_audit(
        args.artifacts_root,
        args.receipts_dir,
        repo_id=args.repo_id,
    )
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
