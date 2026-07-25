import json
from pathlib import Path

from ternary_llm.report import summarize_run


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
