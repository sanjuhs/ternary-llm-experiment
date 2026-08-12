"""Run identical recorded prompts through BF16 and super-ternary endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from modal_minicpm_omni.smoke_test import (
    DEFAULT_SYSTEM_PROMPT,
    check_duplex,
    check_routes,
    read_chunks,
)

DEFAULT_CONTROL = "https://sanjuhs123--minicpm-omni-demo.modal.run"
DEFAULT_CANDIDATE = "https://sanjuhs123--minicpm-omni-super-ternary.modal.run"


def _response_text(report: dict[str, Any]) -> str:
    return "".join(
        str(row.get("text") or "")
        for row in report.get("results", [])
        if row.get("is_listen") is False
    ).strip()


def _run_endpoint(
    *,
    name: str,
    base_url: str,
    prompt_id: str,
    audio_path: Path,
    output_dir: Path,
    units: int,
    system_prompt: str,
) -> dict[str, Any]:
    started = time.monotonic()
    duplex = asyncio.run(
        check_duplex(
            base_url,
            read_chunks(audio_path, units),
            system_prompt=system_prompt,
            audio_output_wav=output_dir / f"{prompt_id}-{name}.wav",
        )
    )
    return {
        "base_url": base_url,
        "routes": check_routes(base_url, ("/health", "/audio_duplex", "/omni")),
        "duplex": duplex,
        "response_text": _response_text(duplex),
        "wall_ms": round((time.monotonic() - started) * 1000, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--audio-f32", type=Path, required=True)
    parser.add_argument("--control-url", default=DEFAULT_CONTROL)
    parser.add_argument("--candidate-url", default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=Path("output/minicpmo_ternary/ab"))
    parser.add_argument("--units", type=int, default=8)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    control = _run_endpoint(
        name="bf16",
        base_url=args.control_url,
        prompt_id=args.prompt_id,
        audio_path=args.audio_f32,
        output_dir=args.output_dir,
        units=args.units,
        system_prompt=args.system_prompt,
    )
    candidate = _run_endpoint(
        name="super-ternary",
        base_url=args.candidate_url,
        prompt_id=args.prompt_id,
        audio_path=args.audio_f32,
        output_dir=args.output_dir,
        units=args.units,
        system_prompt=args.system_prompt,
    )
    similarity = SequenceMatcher(None, control["response_text"], candidate["response_text"]).ratio()
    report = {
        "format": "minicpmo-voice-ab-v1",
        "prompt_id": args.prompt_id,
        "audio_f32": str(args.audio_f32),
        "control": control,
        "candidate": candidate,
        "comparison": {
            "subtitle_sequence_similarity": similarity,
            "warning": (
                "Subtitle similarity is descriptive only. Listen to both WAV files and score "
                "intelligibility, relevance, persona, prosody, and artifacts blind."
            ),
        },
    }
    output_path = args.output_dir / f"{args.prompt_id}-comparison.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output_path), **report}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
