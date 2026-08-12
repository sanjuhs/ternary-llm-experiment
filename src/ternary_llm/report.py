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


def validation_trajectory(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract compact, blog-ready validation diagnostics from metric rows."""
    trajectory = []
    for metric in metrics:
        if metric.get("type") != "validation":
            continue
        attention = metric.get("attention", {}).get("aggregate", {})
        residual = metric.get("residual", {}).get("aggregate", {})
        weight_codes = metric.get("weight_codes", {}).get("fractions", {})
        trajectory.append(
            {
                "step": int(metric["step"]),
                "tokens": int(metric.get("tokens", 0)),
                "loss": float(metric["loss"]),
                "perplexity": float(metric["perplexity"]),
                "attention_zero_fraction": attention.get(
                    "probability_code_0_fraction"
                ),
                "attention_entropy": attention.get("entropy"),
                "residual_normalized_mse": residual.get("normalized_mse"),
                "weight_zero_fraction": weight_codes.get("0"),
            }
        )
    return trajectory


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
        trajectory = validation_trajectory(metrics)
        result["latest_validation_loss"] = validation[-1]["loss"]
        result["latest_validation_perplexity"] = validation[-1]["perplexity"]
        result["validation_trajectory"] = trajectory
        if len(trajectory) >= 2:
            result["validation_loss_change"] = (
                trajectory[-1]["loss"] - trajectory[0]["loss"]
            )
    return result


def _format_optional(value: Any, *, percentage: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    if percentage:
        return f"{100 * value:.2f}%"
    return f"{value:.4f}"


def render_trajectory_markdown(summaries: list[dict[str, Any]]) -> str:
    """Render deterministic Markdown tables for research notes and PRs."""
    lines = ["# Validation trajectories", ""]
    for summary in summaries:
        trajectory = summary.get("validation_trajectory", [])
        if not trajectory:
            continue
        lines.extend(
            [
                f"## {summary['run']}",
                "",
                "| Step | Loss | Perplexity | Zero routes | Attention entropy | Residual NMSE |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in trajectory:
            lines.append(
                "| "
                f"{row['step']:,} | "
                f"{row['loss']:.6f} | "
                f"{row['perplexity']:.4f} | "
                f"{_format_optional(row['attention_zero_fraction'], percentage=True)} | "
                f"{_format_optional(row['attention_entropy'])} | "
                f"{_format_optional(row['residual_normalized_mse'], percentage=True)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--hourly-cost", type=float, required=True)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
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
    if args.format == "markdown":
        print(render_trajectory_markdown(summaries), end="")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
