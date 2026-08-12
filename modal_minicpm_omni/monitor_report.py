"""Audit a JSONL keepalive history against a continuous monitoring window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--minimum-hours", type=float, default=5.0)
    parser.add_argument(
        "--maximum-gap-seconds",
        type=float,
        default=179.0,
        help="Maximum warm-check gap; default leaves 1s inside the 180s scale-down window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("No monitoring records found.")

    records.sort(key=lambda row: float(row["checked_at_unix"]))
    timestamps = [float(row["checked_at_unix"]) for row in records]
    gaps = [later - earlier for earlier, later in zip(timestamps, timestamps[1:], strict=True)]
    span_seconds = timestamps[-1] - timestamps[0]
    failures = [row for row in records if row.get("ok") is not True]
    max_gap = max(gaps, default=0.0)
    report = {
        "records": len(records),
        "successful_records": len(records) - len(failures),
        "failed_records": len(failures),
        "first_checked_at_unix": timestamps[0],
        "last_checked_at_unix": timestamps[-1],
        "span_hours": round(span_seconds / 3600, 3),
        "maximum_gap_seconds": round(max_gap, 1),
        "minimum_hours_required": args.minimum_hours,
        "maximum_gap_seconds_allowed": args.maximum_gap_seconds,
        "duration_met": span_seconds >= args.minimum_hours * 3600,
        "continuity_met": max_gap <= args.maximum_gap_seconds,
        "all_checks_passed": not failures,
    }
    print(json.dumps(report, indent=2))
    if not all((report["duration_met"], report["continuity_met"], report["all_checks_passed"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
