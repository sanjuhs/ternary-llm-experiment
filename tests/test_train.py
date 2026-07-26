import math
from pathlib import Path

from ternary_llm.train import best_logged_validation_loss


def test_best_logged_validation_loss_uses_lowest_finite_value(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"type":"train","loss":3.0}\n'
        '{"type":"validation","loss":2.4}\n'
        '{"type":"validation","loss":2.1}\n'
        '{"type":"validation","loss":2.3}\n'
        "truncated-json\n"
    )

    assert best_logged_validation_loss(metrics) == 2.1


def test_best_logged_validation_loss_defaults_to_infinity(
    tmp_path: Path,
) -> None:
    assert math.isinf(best_logged_validation_loss(tmp_path / "missing.jsonl"))
