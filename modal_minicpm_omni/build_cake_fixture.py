"""Build a 600-second, 60-question cake conversation fixture with macOS TTS."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


SAMPLE_RATE = 16_000
SAMPLE_BYTES = 4
SLOT_SECONDS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("modal_minicpm_omni/cake_questions.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/minicpmo_ternary/long-session/cake-60q-600s.f32"),
    )
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--rate", type=int, default=205)
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if len(questions) != 60:
        raise ValueError(f"Expected 60 questions, got {len(questions)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    slot_bytes = SAMPLE_RATE * SAMPLE_BYTES * SLOT_SECONDS
    leading_silence = bytes(SAMPLE_RATE * SAMPLE_BYTES // 4)
    manifest = []

    with tempfile.TemporaryDirectory(prefix="minicpm-cake-fixture-") as tmp:
        tmp_dir = Path(tmp)
        with args.output.open("wb") as output:
            for index, row in enumerate(questions):
                aiff = tmp_dir / f"question-{index:02d}.aiff"
                raw = tmp_dir / f"question-{index:02d}.f32"
                subprocess.run(
                    ["say", "-v", args.voice, "-r", str(args.rate), "-o", str(aiff), row["question"]],
                    check=True,
                )
                subprocess.run(
                    [
                        "ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff),
                        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", str(raw),
                    ],
                    check=True,
                )
                speech = raw.read_bytes()
                usable = max(0, slot_bytes - len(leading_silence))
                slot = (leading_silence + speech[:usable]).ljust(slot_bytes, b"\0")
                output.write(slot)
                manifest.append(
                    {
                        **row,
                        "question_index": index + 1,
                        "starts_at_s": index * SLOT_SECONDS,
                        "speech_s": round(min(len(speech), usable) / SAMPLE_RATE / SAMPLE_BYTES, 3),
                    }
                )

    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "audio": str(args.output),
        "manifest": str(manifest_path),
        "questions": len(manifest),
        "duration_s": len(manifest) * SLOT_SECONDS,
        "bytes": args.output.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
