from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_metrics(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def summarize_run(
    metrics_path: Path,
    *,
    token_budget: int,
    hourly_cost: float,
) -> dict[str, Any]:
    metrics = read_metrics(metrics_path)
    training = [metric for metric in metrics if metric.get("type") == "train"]
    validation = [metric for metric in metrics if metric.get("type") == "validation"]
    if not training:
        raise ValueError(f"no training metrics in {metrics_path}")

    throughput_samples = [
        float(metric["tokens_per_second"])
        for metric in training[1:]
        if float(metric["tokens_per_second"]) > 0
    ]
    if not throughput_samples:
        throughput_samples = [float(training[-1]["tokens_per_second"])]
    throughput = statistics.median(throughput_samples)
    hours = token_budget / throughput / 3600
    result: dict[str, Any] = {
        "run": metrics_path.parent.name,
        "measured_steps": int(training[-1]["step"]),
        "median_tokens_per_second": throughput,
        "token_budget": token_budget,
        "estimated_hours": hours,
        "estimated_compute_cost": hours * hourly_cost,
    }
    if validation:
        result["latest_validation_loss"] = validation[-1]["loss"]
        result["latest_validation_perplexity"] = validation[-1]["perplexity"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--hourly-cost", type=float, required=True)
    args = parser.parse_args()

    summaries = [
        summarize_run(
            path,
            token_budget=args.token_budget,
            hourly_cost=args.hourly_cost,
        )
        for path in sorted(args.artifacts.glob("*/metrics.jsonl"))
    ]
    if not summaries:
        raise FileNotFoundError(f"no */metrics.jsonl files under {args.artifacts}")
    output = {
        "runs": summaries,
        "total_estimated_hours": sum(item["estimated_hours"] for item in summaries),
        "total_estimated_compute_cost": sum(item["estimated_compute_cost"] for item in summaries),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
