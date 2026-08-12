"""Health and duplex smoke test for the public MiniCPM-o demo.

Examples:
    python modal_minicpm_omni/smoke_test.py
    python modal_minicpm_omni/smoke_test.py --audio-f32 /tmp/sample.f32 --units 8

The optional audio file must be mono float32 PCM at 16 kHz.  The script waits
for each result only when benchmarking; the browser client streams units on a
fixed one-second clock and therefore doesn't serialize network round trips.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import base64
import json
import ssl
import statistics
import sys
import time
import urllib.request
import uuid
import wave
from pathlib import Path
from urllib.parse import urlparse

import certifi
import websockets

DEFAULT_BASE_URL = "https://sanjuhs123--minicpm-omni-demo.modal.run"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, natural English voice assistant."
CHUNK_BYTES = 16_000 * 4  # one second of mono 16 kHz float32 PCM


def ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def check_routes(base_url: str, paths: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    context = ssl_context()
    for path in paths:
        started = time.monotonic()
        request = urllib.request.Request(
            base_url.rstrip("/") + path,
            headers={"User-Agent": "minicpm-modal-smoke/1"},
        )
        with urllib.request.urlopen(request, timeout=180, context=context) as response:
            body = response.read()
            rows.append(
                {
                    "path": path,
                    "status": response.status,
                    "bytes": len(body),
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                }
            )
    return rows


def websocket_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    session_id = f"adx_smoke_{uuid.uuid4().hex[:12]}"
    return f"{scheme}://{parsed.netloc}/ws/duplex/{session_id}"


async def check_duplex(
    base_url: str,
    audio_chunks: list[bytes],
    origin: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    audio_output_wav: Path | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    events: list[str] = []
    results: list[dict[str, object]] = []
    generated_audio = bytearray()

    async with websockets.connect(
        websocket_url(base_url),
        ssl=ssl_context(),
        open_timeout=30,
        close_timeout=10,
        max_size=8 * 1024 * 1024,
        origin=origin,
    ) as socket:
        prepare_sent = False
        while True:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=90))
            kind = message.get("type")
            if isinstance(kind, str):
                events.append(kind)
            if kind == "queue_done" and not prepare_sent:
                await socket.send(
                    json.dumps(
                        {
                            "type": "prepare",
                            "system_prompt": system_prompt,
                            "config": {"length_penalty": 1.05},
                        }
                    )
                )
                prepare_sent = True
            elif kind == "prepared":
                break
            elif kind == "error":
                raise RuntimeError(message.get("error") or message)

        for index, chunk in enumerate(audio_chunks, start=1):
            unit_started = time.monotonic()
            await socket.send(
                json.dumps(
                    {
                        "type": "audio_chunk",
                        "audio_base64": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )
            while True:
                message = json.loads(await asyncio.wait_for(socket.recv(), timeout=90))
                if message.get("type") == "result":
                    break
                if message.get("type") == "error":
                    raise RuntimeError(message.get("error") or message)
            results.append(
                {
                    "unit": index,
                    "wall_ms": round((time.monotonic() - unit_started) * 1000, 1),
                    "cost_all_ms": message.get("cost_all_ms"),
                    "cost_llm_ms": message.get("cost_llm_ms"),
                    "cost_tts_ms": message.get("cost_tts_ms"),
                    "is_listen": message.get("is_listen"),
                    "n_tokens": message.get("n_tokens"),
                    "text": (message.get("text") or "")[:120],
                }
            )
            encoded_audio = message.get("audio_data")
            if isinstance(encoded_audio, str) and encoded_audio:
                generated_audio.extend(base64.b64decode(encoded_audio))

        await socket.send(json.dumps({"type": "stop"}))
        while True:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
            kind = message.get("type")
            if isinstance(kind, str):
                events.append(kind)
            if kind == "stopped":
                break
            if kind == "error":
                raise RuntimeError(message.get("error") or message)

    costs = [
        row["cost_all_ms"] for row in results if isinstance(row.get("cost_all_ms"), (int, float))
    ]
    speaking_costs = [
        row["cost_all_ms"]
        for row in results
        if row.get("is_listen") is False and isinstance(row.get("cost_all_ms"), (int, float))
    ]
    if audio_output_wav is not None and generated_audio:
        write_float32_wav(audio_output_wav, bytes(generated_audio), sample_rate=24_000)
    return {
        "events": events,
        "prepared": "prepared" in events,
        "stopped": "stopped" in events,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        "results": results,
        "server_mean_ms": round(statistics.mean(costs), 1) if costs else None,
        "server_max_ms": round(max(costs), 1) if costs else None,
        "speaking_mean_ms": (round(statistics.mean(speaking_costs), 1) if speaking_costs else None),
        "all_units_under_1s": all(value < 1000 for value in costs) if costs else None,
        "generated_audio_samples": len(generated_audio) // 4,
        "audio_output_wav": str(audio_output_wav) if generated_audio else None,
    }


def write_float32_wav(path: Path, payload: bytes, *, sample_rate: int) -> None:
    """Write little-endian float32 PCM returned by MiniCPM-o as PCM16 WAV."""

    samples = array.array("f")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    pcm = array.array(
        "h",
        (round(max(-1.0, min(1.0, sample)) * 32767) for sample in samples),
    )
    if sys.byteorder != "little":
        pcm.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def read_chunks(path: Path | None, units: int) -> list[bytes]:
    if path is None:
        return []
    payload = path.read_bytes()
    return [
        payload[offset : offset + CHUNK_BYTES]
        for offset in range(0, min(len(payload), units * CHUNK_BYTES), CHUNK_BYTES)
        if len(payload[offset : offset + CHUNK_BYTES]) == CHUNK_BYTES
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--audio-f32", type=Path)
    parser.add_argument("--units", type=int, default=8)
    parser.add_argument(
        "--origin",
        help="Send a browser-style Origin header when validating a hosted frontend.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt sent during the duplex prepare handshake.",
    )
    parser.add_argument(
        "--routes-only",
        action="store_true",
        help="Avoid claiming the single duplex worker; useful for frequent keepalives.",
    )
    parser.add_argument(
        "--record-jsonl",
        type=Path,
        help="Append the timestamped result to a JSONL monitoring history.",
    )
    parser.add_argument(
        "--audio-output-wav",
        type=Path,
        help="Save the model's generated 24 kHz audio as a PCM16 WAV file.",
    )
    return parser.parse_args()


def emit_report(report: dict[str, object], record_path: Path | None) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {
        "base_url": args.base_url,
        "checked_at_unix": time.time(),
    }
    try:
        report["routes"] = check_routes(args.base_url, ("/health", "/audio_duplex", "/omni"))
        if not args.routes_only:
            report["duplex"] = asyncio.run(
                check_duplex(
                    args.base_url,
                    read_chunks(args.audio_f32, args.units),
                    origin=args.origin,
                    system_prompt=args.system_prompt,
                    audio_output_wav=args.audio_output_wav,
                )
            )
        routes = report["routes"]
        assert isinstance(routes, list)
        assert all(row["status"] == 200 for row in routes)
        if not args.routes_only:
            duplex = report["duplex"]
            assert isinstance(duplex, dict)
            assert duplex["prepared"] and duplex["stopped"]
            if args.audio_f32 is not None:
                assert duplex["all_units_under_1s"] is True
        report["ok"] = True
    except Exception as error:
        report["ok"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        emit_report(report, args.record_jsonl)
        raise
    emit_report(report, args.record_jsonl)


if __name__ == "__main__":
    main()
