from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ternary_llm.artifacts import build_run_manifest


class OvernightSummaryError(ValueError):
    """Raised when the overnight experiment chain is incomplete or inconsistent."""


ManifestBuilder = Callable[..., dict[str, Any]]


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OvernightSummaryError(f"required result is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise OvernightSummaryError(f"result is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise OvernightSummaryError(f"result must contain a JSON object: {path}")
    return payload


def _require_success(path: Path) -> str:
    marker = path / "SUCCESS"
    if not marker.is_file() or not marker.read_text(encoding="utf-8").strip():
        raise OvernightSummaryError(f"stage has no non-empty SUCCESS marker: {path}")
    return marker.read_text(encoding="utf-8").strip()


def _source_checkpoint(path: Path, artifacts_root: Path) -> str:
    source_path = path / "source-checkpoint.txt"
    if not source_path.is_file():
        raise OvernightSummaryError(f"stage has no source-checkpoint.txt: {path}")
    declared = Path(source_path.read_text(encoding="utf-8").strip())
    if not declared.is_absolute():
        declared = artifacts_root.parent.parent / declared
    if not declared.is_file():
        raise OvernightSummaryError(f"declared source checkpoint does not exist: {declared}")
    return str(declared)


def _run_names(artifacts_root: Path) -> tuple[list[str], dict[str, Any]]:
    clip = _json_object(artifacts_root / "attention-clip-refinement" / "selection.json")
    scale = _json_object(artifacts_root / "shared-qkv-scale-refinement" / "selection.json")
    softmax = _json_object(artifacts_root / "softmax1-refinement" / "comparison.json")
    relu = _json_object(artifacts_root / "relu-hardening" / "comparison.json")
    binary = _json_object(artifacts_root / "binary-qk-fallback" / "comparison.json")
    rms_norm = _json_object(
        artifacts_root / "integer-rmsnorm-screen" / "comparison.json"
    )
    strict_contract = _json_object(
        artifacts_root / "strict-contract-refinement" / "comparison.json"
    )

    selected_clip = str(clip.get("selected_clip", ""))
    selected_scale = str(scale.get("selected_initial_scale", ""))
    if not selected_clip or not selected_scale:
        raise OvernightSummaryError("clip and QKV-scale selections must be non-empty")

    return (
        [
            "hadamard-ternary-p3-half-pass",
            "hadamard-ternary-p3-step-10000-preserved",
            f"attention-clip-selected-{selected_clip}-refine",
            "qkv-scale-token-control",
            f"shared-head-qkv-scale-{selected_scale}-refine",
            "softmax-control-refine",
            "softmax1-no-update-refine",
            "ffn-gelu-control-refine",
            "ffn-relu-harden-refine",
            "binary-qk-ternary-control-refine",
            "binary-qk-sign-distill-refine",
            "strict-contract-relu-rmsnorm",
        ],
        {
            "attention_clip": clip,
            "qkv_scale": scale,
            "attention_normalization": softmax,
            "feed_forward_activation": relu,
            "qkv_quantization": binary,
            "rms_norm_quantization": rms_norm,
            "strict_operand_contract": strict_contract,
        },
    )


def build_overnight_summary(
    artifacts_root: Path,
    *,
    manifest_builder: ManifestBuilder = build_run_manifest,
) -> dict[str, Any]:
    """Build a fail-closed summary of the completed overnight experiment chain."""
    artifacts_root = artifacts_root.resolve()
    stages = [
        "hadamard-ternary-p3-half-pass",
        "hadamard-ternary-p3-step-10000-preserved",
        "attention-clip-refinement",
        "shared-qkv-scale-refinement",
        "softmax1-refinement",
        "relu-hardening",
        "binary-qk-fallback",
        "integer-rmsnorm-screen",
        "strict-contract-refinement",
    ]
    success = {stage: _require_success(artifacts_root / stage) for stage in stages}
    lineage_stages = stages[2:]
    lineage = {
        stage: _source_checkpoint(artifacts_root / stage, artifacts_root)
        for stage in lineage_stages
    }
    run_names, decisions = _run_names(artifacts_root)

    manifests = []
    for name in run_names:
        run_dir = artifacts_root / name
        manifest = manifest_builder(
            run_dir,
            require_complete_checksums=True,
        )
        if manifest.get("run") != name:
            raise OvernightSummaryError(
                f"manifest run mismatch: expected {name}, got {manifest.get('run')}"
            )
        manifests.append(manifest)

    ranked = sorted(
        (
            {
                "run": manifest["run"],
                "loss": float(manifest["validation"]["loss"]),
                "perplexity": float(manifest["validation"]["perplexity"]),
                "evaluated_tokens": int(manifest["validation"]["evaluated_tokens"]),
            }
            for manifest in manifests
        ),
        key=lambda item: (item["loss"], item["run"]),
    )
    return {
        "status": "complete",
        "selection_rule": "lowest matched exhaustive sequential validation loss",
        "best_run": ranked[0],
        "ranking": ranked,
        "decisions": decisions,
        "lineage": lineage,
        "success_markers": success,
        "audited_runs": manifests,
    }


def render_overnight_markdown(summary: dict[str, Any]) -> str:
    best = summary["best_run"]
    lines = [
        "# Overnight ternary experiment summary",
        "",
        (
            f"Best matched run: **{best['run']}** with validation loss "
            f"**{best['loss']:.6f}** and perplexity **{best['perplexity']:.4f}**."
        ),
        "",
        "| Rank | Run | Validation loss | Perplexity | Evaluated tokens |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary["ranking"], start=1):
        lines.append(
            f"| {rank} | {row['run']} | {row['loss']:.6f} | "
            f"{row['perplexity']:.4f} | {row['evaluated_tokens']:,} |"
        )
    lines.extend(
        [
            "",
            "Every listed run passed its artifact audit and complete SHA-256 check.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts_root", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    summary = build_overnight_summary(args.artifacts_root)
    markdown = render_overnight_markdown(summary)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown, end="")


if __name__ == "__main__":
    main()
