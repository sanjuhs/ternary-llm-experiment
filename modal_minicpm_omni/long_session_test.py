"""Run a paced 10-minute, 60-question full-duplex cake conversation.

This is a transport, context-window, end-of-turn, and coarse relevance harness.
It intentionally sends one audio unit per wall-clock second, like the browser.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path
from typing import Any

import websockets

if __package__:
    from .smoke_test import (
        BINARY_AUDIO_MAGIC,
        CHUNK_BYTES,
        MAX_SPEAKING_UNITS,
        check_routes,
        ssl_context,
        websocket_url,
        write_float32_wav,
    )
else:
    from smoke_test import (  # type: ignore[no-redef]
        BINARY_AUDIO_MAGIC,
        CHUNK_BYTES,
        MAX_SPEAKING_UNITS,
        check_routes,
        ssl_context,
        websocket_url,
        write_float32_wav,
    )


DEFAULT_GUARDED = "https://sanjuhs123--minicpm-omni-guarded-ternary.modal.run"
DEFAULT_PROMPT = """You are gaMMA, a warm 54-year-old Swiss-German master baker.
You run bakeries in Switzerland, France, and Bangalore and plan one in Shanghai.
You love chocolate truffles and Chocolate Extreme brownies. Answer each cake
question directly in no more than two short sentences, normally under eight
seconds of speech. Do not repeat yourself. End each answer cleanly, then listen."""


def load_units(path: Path, count: int) -> list[bytes]:
    payload = path.read_bytes()
    expected = count * CHUNK_BYTES
    if len(payload) != expected:
        raise ValueError(f"Expected {expected} bytes ({count}s), got {len(payload)}")
    return [payload[index * CHUNK_BYTES : (index + 1) * CHUNK_BYTES] for index in range(count)]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return round(ordered[index], 1)


async def run_session(
    *,
    base_url: str,
    units: list[bytes],
    system_prompt: str,
) -> dict[str, Any]:
    events: list[str] = []
    results: list[dict[str, Any]] = []
    generated_audio = bytearray()
    result_ready = asyncio.Event()
    stopped = asyncio.Event()
    state = {"consecutive_speaking": 0, "binary_audio_v1": False, "forced_units": 0}

    async with websockets.connect(
        websocket_url(base_url),
        ssl=ssl_context(),
        open_timeout=60,
        close_timeout=15,
        max_size=16 * 1024 * 1024,
    ) as socket:
        while True:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=180))
            kind = message.get("type")
            if isinstance(kind, str):
                events.append(kind)
            if kind == "queue_done":
                state["binary_audio_v1"] = bool(
                    (message.get("capabilities") or {}).get("binary_audio_v1")
                )
                await socket.send(json.dumps({
                    "type": "prepare",
                    "system_prompt": system_prompt,
                    "config": {"length_penalty": 0.95},
                }))
            elif kind == "prepared":
                break
            elif kind == "error":
                raise RuntimeError(message.get("error") or message)

        session_start = time.monotonic()

        async def receive_results() -> None:
            while not stopped.is_set():
                message = json.loads(await asyncio.wait_for(socket.recv(), timeout=45))
                kind = message.get("type")
                if kind == "error":
                    raise RuntimeError(message.get("error") or message)
                if kind == "stopped":
                    events.append("stopped")
                    stopped.set()
                    return
                if kind != "result":
                    continue

                received_at = time.monotonic() - session_start
                is_listen = bool(message.get("is_listen", True))
                if is_listen:
                    state["consecutive_speaking"] = 0
                else:
                    state["consecutive_speaking"] += 1
                encoded = message.get("audio_data")
                if isinstance(encoded, str) and encoded:
                    generated_audio.extend(base64.b64decode(encoded))
                results.append({
                    "result_index": len(results) + 1,
                    "received_at_s": round(received_at, 3),
                    "is_listen": is_listen,
                    "end_of_turn": bool(message.get("end_of_turn")),
                    "text": message.get("text") or "",
                    "current_time": message.get("current_time"),
                    "n_tokens": message.get("n_tokens"),
                    "cost_all_ms": message.get("cost_all_ms"),
                    "wall_clock_ms": message.get("wall_clock_ms"),
                    "kv_cache_length": message.get("kv_cache_length"),
                })
                if len(results) >= len(units):
                    result_ready.set()

        receiver = asyncio.create_task(receive_results())
        send_lateness_ms: list[float] = []
        try:
            for index, chunk in enumerate(units):
                target = session_start + index
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                send_lateness_ms.append(max(0.0, (time.monotonic() - target) * 1000))

                force_listen = state["consecutive_speaking"] >= MAX_SPEAKING_UNITS
                if force_listen:
                    state["forced_units"] += 1
                if state["binary_audio_v1"]:
                    flags = 1 if force_listen else 0
                    await socket.send(BINARY_AUDIO_MAGIC + bytes((1, flags, 0, 0)) + chunk)
                else:
                    await socket.send(json.dumps({
                        "type": "audio_chunk",
                        "audio_base64": base64.b64encode(chunk).decode("ascii"),
                        "force_listen": force_listen,
                    }))

                if index and index % 15 == 0:
                    await socket.send(json.dumps({
                        "type": "client_diagnostic",
                        "event": "long_session_heartbeat",
                        "unit": index + 1,
                        "sent_at_unix": time.time(),
                    }))
                if (index + 1) % 60 == 0:
                    print(json.dumps({
                        "progress_units": index + 1,
                        "progress_seconds": index + 1,
                        "results_received": len(results),
                        "consecutive_speaking": state["consecutive_speaking"],
                    }), flush=True)

            await asyncio.wait_for(result_ready.wait(), timeout=180)
            await socket.send(json.dumps({"type": "stop"}))
            await asyncio.wait_for(stopped.wait(), timeout=30)
            await receiver
        finally:
            if not receiver.done():
                receiver.cancel()

    return {
        "events": events,
        "binary_audio_v1": state["binary_audio_v1"],
        "forced_listen_units": state["forced_units"],
        "elapsed_s": round(time.monotonic() - session_start, 3),
        "send_lateness_ms": send_lateness_ms,
        "results": results,
        "generated_audio": bytes(generated_audio),
    }


def summarize(
    session: dict[str, Any],
    questions: list[dict[str, Any]],
    expected_units: int,
) -> dict[str, Any]:
    results = session["results"]
    costs = [float(row["cost_all_ms"]) for row in results if isinstance(row.get("cost_all_ms"), (int, float))]
    kv = [int(row["kv_cache_length"]) for row in results if isinstance(row.get("kv_cache_length"), int)]
    arrivals = [float(row["received_at_s"]) for row in results]
    gaps = [right - left for left, right in zip(arrivals, arrivals[1:])]
    # Context mode drops one old unit at a time after the window fills.  Because
    # a new unit is added in the same cycle, the visible KV decrease is often
    # only 1–5 tokens rather than one large sawtooth reset.
    sliding_events = sum(1 for left, right in zip(kv, kv[1:]) if right < left)

    max_speaking = 0
    speaking_streak = 0
    turns: list[str] = []
    turn_text = ""
    slot_text = ["" for _ in questions]
    for row in results:
        if row["is_listen"]:
            if turn_text.strip():
                turns.append(turn_text.strip())
                turn_text = ""
            speaking_streak = 0
            continue
        speaking_streak += 1
        max_speaking = max(max_speaking, speaking_streak)
        text = str(row.get("text") or "")
        turn_text += text
        current_time = row.get("current_time")
        if isinstance(current_time, int):
            slot = max(0, current_time - 1) // 10
            if slot < len(slot_text):
                slot_text[slot] += text
        if row.get("end_of_turn") and turn_text.strip():
            turns.append(turn_text.strip())
            turn_text = ""
            speaking_streak = 0
    if turn_text.strip():
        turns.append(turn_text.strip())

    coverage = sum(bool(text.strip()) for text in slot_text)
    keyword_hits = 0
    per_question = []
    for row, response in zip(questions, slot_text):
        lowered = response.lower()
        matched = [word for word in row["keywords"] if word.lower() in lowered]
        if matched:
            keyword_hits += 1
        per_question.append({
            "question_index": row["question_index"],
            "question": row["question"],
            "response": response.strip(),
            "matched_keywords": matched,
        })

    duplicates = len(turns) - len({" ".join(turn.lower().split()) for turn in turns})
    operational_pass = (
        len(results) == expected_units
        and (max(gaps, default=0) <= 5.0)
        and max_speaking <= MAX_SPEAKING_UNITS
        and (max(kv, default=0) < 10_000)
        and sliding_events >= 1
    )
    quality_signal_pass = coverage >= 42 and keyword_hits >= 24 and duplicates <= 3
    return {
        "operational_pass": operational_pass,
        "quality_signal_pass": quality_signal_pass,
        "result_units": len(results),
        "expected_units": expected_units,
        "binary_audio_v1": session["binary_audio_v1"],
        "elapsed_s": session["elapsed_s"],
        "forced_listen_units": session["forced_listen_units"],
        "max_consecutive_speaking_units": max_speaking,
        "completed_text_turns": len(turns),
        "exact_duplicate_turns": duplicates,
        "question_slots_with_response": coverage,
        "question_slots_with_keyword_hit": keyword_hits,
        "server_ms": {
            "mean": round(statistics.mean(costs), 1) if costs else None,
            "p50": percentile(costs, 0.5),
            "p95": percentile(costs, 0.95),
            "max": round(max(costs), 1) if costs else None,
            "under_realtime_fraction": round(sum(cost <= 1000 for cost in costs) / len(costs), 4) if costs else None,
        },
        "transport": {
            "max_result_gap_s": round(max(gaps), 3) if gaps else None,
            "p95_send_lateness_ms": percentile(session["send_lateness_ms"], 0.95),
        },
        "context": {
            "first_kv_tokens": kv[0] if kv else None,
            "peak_kv_tokens": max(kv) if kv else None,
            "last_kv_tokens": kv[-1] if kv else None,
            "sliding_events": sliding_events,
        },
        "per_question": per_question,
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_GUARDED)
    parser.add_argument("--label", default="guarded-ternary")
    parser.add_argument(
        "--audio-f32",
        type=Path,
        default=Path("output/minicpmo_ternary/long-session/cake-60q-600s.f32"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/minicpmo_ternary/long-session/cake-60q-600s.json"),
    )
    parser.add_argument("--system-prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/minicpmo_ternary/long-session/results"),
    )
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    questions = json.loads(args.manifest.read_text(encoding="utf-8"))
    units = load_units(args.audio_f32, len(questions) * 10)
    routes = check_routes(args.base_url, ("/health", "/audio_duplex", "/omni"))
    session = asyncio.run(run_session(
        base_url=args.base_url,
        units=units,
        system_prompt=args.system_prompt,
    ))
    summary = summarize(session, questions, len(units))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = args.output_dir / f"{args.label}-cake-10m.wav"
    if session["generated_audio"]:
        write_float32_wav(wav_path, session["generated_audio"], sample_rate=24_000)
    report = {
        "format": "minicpmo-cake-long-session-v1",
        "label": args.label,
        "base_url": args.base_url,
        "checked_at_unix": time.time(),
        "routes": routes,
        "summary": summary,
        "results": session["results"],
        "audio_output_wav": str(wav_path) if session["generated_audio"] else None,
    }
    report_path = args.output_dir / f"{args.label}-cake-10m.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(report_path), **summary}, indent=2, ensure_ascii=False))
    if not args.no_fail and not (summary["operational_pass"] and summary["quality_signal_pass"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
