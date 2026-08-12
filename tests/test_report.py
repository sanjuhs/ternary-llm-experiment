import json
from pathlib import Path

from ternary_llm.report import render_trajectory_markdown, summarize_run


def test_summarize_run_uses_steady_state_median(tmp_path: Path) -> None:
    metrics = [
        {"type": "train", "step": 1, "tokens_per_second": 10},
        {"type": "train", "step": 10, "tokens_per_second": 1000},
        {"type": "train", "step": 20, "tokens_per_second": 1200},
        {"type": "validation", "step": 20, "loss": 2.0, "perplexity": 7.4},
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "".join(json.dumps(metric) + "\n" for metric in metrics),
        encoding="utf-8",
    )
    result = summarize_run(path, token_budget=3_960_000, hourly_cost=0.5)

    assert result["median_tokens_per_second"] == 1100
    assert result["estimated_hours"] == 1
    assert result["estimated_compute_cost"] == 0.5
    assert result["latest_validation_loss"] == 2.0


def test_summarize_run_extracts_diagnostic_trajectory_markdown(
    tmp_path: Path,
) -> None:
    metrics = [
        {"type": "train", "step": 1, "tokens_per_second": 1000},
        {
            "type": "validation",
            "step": 10,
            "tokens": 100,
            "loss": 2.4,
            "perplexity": 11.0,
            "attention": {
                "aggregate": {
                    "probability_code_0_fraction": 0.8,
                    "entropy": 2.3,
                }
            },
            "residual": {"aggregate": {"normalized_mse": 0.017}},
            "weight_codes": {"fractions": {"0": 0.34}},
        },
        {
            "type": "validation",
            "step": 20,
            "tokens": 200,
            "loss": 2.3,
            "perplexity": 10.0,
            "attention": {
                "aggregate": {
                    "probability_code_0_fraction": 0.85,
                    "entropy": 2.1,
                }
            },
            "residual": {"aggregate": {"normalized_mse": 0.018}},
            "weight_codes": {"fractions": {"0": 0.33}},
        },
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "".join(json.dumps(metric) + "\n" for metric in metrics),
        encoding="utf-8",
    )

    summary = summarize_run(path, token_budget=3_600_000, hourly_cost=0.5)
    markdown = render_trajectory_markdown([summary])

    assert abs(summary["validation_loss_change"] + 0.1) < 1e-12
    assert summary["validation_trajectory"][-1]["attention_zero_fraction"] == 0.85
    assert "| 20 | 2.300000 | 10.0000 | 85.00% | 2.1000 | 1.80% |" in markdown
