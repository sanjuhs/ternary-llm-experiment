from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modal_minicpm_omni.long_session_test import summarize
from modal_minicpm_omni.patches.patch_duplex_reliability import replace_once
from modal_minicpm_omni.smoke_test import (
    BINARY_AUDIO_HEADER_BYTES,
    BINARY_AUDIO_MAGIC,
    CHUNK_BYTES,
)

def test_cake_fixture_contract_has_sixty_distinct_questions() -> None:
    rows = json.loads((ROOT / "modal_minicpm_omni/cake_questions.json").read_text())
    assert len(rows) == 60
    assert len({row["question"] for row in rows}) == 60
    assert all(row["keywords"] for row in rows)


def test_binary_audio_v1_is_one_second_float32_pcm() -> None:
    frame = BINARY_AUDIO_MAGIC + bytes((1, 0, 0, 0)) + bytes(CHUNK_BYTES)
    assert frame[:4] == b"MCPM"
    assert len(frame) == BINARY_AUDIO_HEADER_BYTES + 16_000 * 4


def test_exact_patch_helper_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "source.py"
    target.write_text("before\n", encoding="utf-8")
    replace_once(target, "before", "after # marker", "# marker")
    replace_once(target, "before", "after # marker", "# marker")
    assert target.read_text(encoding="utf-8") == "after # marker\n"


def test_long_session_summary_detects_healthy_windowed_run() -> None:
    questions = json.loads((ROOT / "modal_minicpm_omni/cake_questions.json").read_text())
    questions = [dict(row, question_index=index + 1) for index, row in enumerate(questions)]
    results = []
    for unit in range(1, 601):
        slot = (unit - 1) // 10
        speaking = (unit - 1) % 10 == 2
        keyword = questions[slot]["keywords"][0] if speaking else ""
        # A sawtooth models successful context compaction every 180 units.
        kv = 1_000 + ((unit - 1) % 180) * 16
        results.append(
            {
                "received_at_s": float(unit),
                "is_listen": not speaking,
                "end_of_turn": speaking,
                "text": f"Answer {slot + 1} discusses {keyword}." if speaking else "",
                "current_time": unit,
                "cost_all_ms": 100.0,
                "kv_cache_length": kv,
            }
        )
    session = {
        "results": results,
        "binary_audio_v1": True,
        "forced_listen_units": 0,
        "elapsed_s": 600.0,
        "send_lateness_ms": [0.0] * 600,
    }
    report = summarize(session, questions, 600)
    assert report["operational_pass"] is True
    assert report["quality_signal_pass"] is True
    assert report["context"]["sliding_events"] >= 3
    assert report["question_slots_with_keyword_hit"] == 60
