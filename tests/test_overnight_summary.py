import json
from pathlib import Path
from typing import Any

import pytest

from ternary_llm.overnight_summary import (
    OvernightSummaryError,
    build_overnight_summary,
    render_overnight_markdown,
)

STAGES = (
    "hadamard-ternary-p3-half-pass",
    "hadamard-ternary-p3-step-10000-preserved",
    "attention-clip-refinement",
    "shared-qkv-scale-refinement",
    "softmax1-refinement",
    "relu-hardening",
    "binary-qk-fallback",
    "integer-rmsnorm-screen",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete_chain(root: Path) -> None:
    source = root / "source.pt"
    source.write_bytes(b"checkpoint")
    for stage in STAGES:
        stage_dir = root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "SUCCESS").write_text("complete\n", encoding="utf-8")
    for stage in STAGES[2:]:
        (root / stage / "source-checkpoint.txt").write_text(
            str(source) + "\n",
            encoding="utf-8",
        )

    _write_json(
        root / "attention-clip-refinement" / "selection.json",
        {"selected_clip": "2.5"},
    )
    _write_json(
        root / "shared-qkv-scale-refinement" / "selection.json",
        {"selected_initial_scale": "1.0"},
    )
    _write_json(
        root / "softmax1-refinement" / "comparison.json",
        {"selected_normalization": "softmax1"},
    )
    _write_json(
        root / "relu-hardening" / "comparison.json",
        {"selected_feed_forward_activation": "relu"},
    )
    _write_json(
        root / "binary-qk-fallback" / "comparison.json",
        {"selected_qkv_quantization": "binary_qk_ternary_v"},
    )
    _write_json(
        root / "integer-rmsnorm-screen" / "comparison.json",
        {"selected_rms_norm_quantization": "integer_reference"},
    )


def test_build_overnight_summary_requires_complete_chain(tmp_path: Path) -> None:
    _complete_chain(tmp_path)
    seen: list[tuple[str, bool]] = []

    def manifest(run_dir: Path, *, require_complete_checksums: bool) -> dict[str, Any]:
        seen.append((run_dir.name, require_complete_checksums))
        rank = len(seen)
        return {
            "run": run_dir.name,
            "validation": {
                "loss": 3.0 - rank / 100,
                "perplexity": 10.0 - rank / 10,
                "evaluated_tokens": 1000,
            },
        }

    summary = build_overnight_summary(tmp_path, manifest_builder=manifest)

    assert len(seen) == 11
    assert all(complete for _, complete in seen)
    assert summary["status"] == "complete"
    assert summary["best_run"]["run"] == "binary-qk-sign-distill-refine"
    assert "attention_clip" in summary["decisions"]
    markdown = render_overnight_markdown(summary)
    assert "# Overnight ternary experiment summary" in markdown
    assert "| 1 | binary-qk-sign-distill-refine |" in markdown


def test_build_overnight_summary_fails_without_success_marker(
    tmp_path: Path,
) -> None:
    _complete_chain(tmp_path)
    (tmp_path / "relu-hardening" / "SUCCESS").unlink()

    with pytest.raises(OvernightSummaryError, match="SUCCESS"):
        build_overnight_summary(tmp_path)


def test_build_overnight_summary_fails_on_missing_lineage_checkpoint(
    tmp_path: Path,
) -> None:
    _complete_chain(tmp_path)
    (tmp_path / "source.pt").unlink()

    with pytest.raises(OvernightSummaryError, match="source checkpoint"):
        build_overnight_summary(tmp_path)
