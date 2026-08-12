"""Harden the pinned MiniCPM-o duplex runtime for long browser sessions.

The patch is deliberately exact-match and idempotent.  If the pinned upstream
revision changes, image construction fails instead of applying a partial patch.
"""

from __future__ import annotations

from pathlib import Path


APP_DIR = Path("/app")


def replace_once(path: Path, needle: str, replacement: str, marker: str) -> None:
    source = path.read_text(encoding="utf-8")
    if marker in source:
        return
    if source.count(needle) != 1:
        raise RuntimeError(f"Pinned upstream source changed at {path}; refusing unsafe patch")
    path.write_text(source.replace(needle, replacement), encoding="utf-8")


def patch_gateway() -> None:
    path = APP_DIR / "gateway.py"
    replace_once(
        path,
        "import os\nimport re\nimport json\n",
        "import os\nimport re\nimport json\nimport base64\n",
        "import json\nimport base64\nimport asyncio",
    )

    replace_once(
        path,
        """def _sync_append(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


# ============ Duplex WebSocket（有状态，FIFO 排队 + 代理到 Worker） ============
""",
        """def _sync_append(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


_BINARY_AUDIO_MAGIC = b"MCPM"
_BINARY_AUDIO_HEADER_BYTES = 8
_BINARY_AUDIO_BYTES_PER_UNIT = 16_000 * 4


def _binary_audio_to_legacy_message(payload: bytes) -> str:
    # Convert browser binary-audio-v1 frames to the stable worker JSON protocol.
    if len(payload) != _BINARY_AUDIO_HEADER_BYTES + _BINARY_AUDIO_BYTES_PER_UNIT:
        raise ValueError(f"invalid binary audio frame length: {len(payload)}")
    if payload[:4] != _BINARY_AUDIO_MAGIC or payload[4] != 1:
        raise ValueError("invalid binary audio frame header")
    force_listen = bool(payload[5] & 1)
    return json.dumps({
        "type": "audio_chunk",
        "audio_base64": base64.b64encode(payload[_BINARY_AUDIO_HEADER_BYTES:]).decode("ascii"),
        "force_listen": force_listen,
    })


# ============ Duplex WebSocket（有状态，FIFO 排队 + 代理到 Worker） ============
""",
        "_BINARY_AUDIO_MAGIC = b\"MCPM\"",
    )

    replace_once(
        path,
        """    await ws.send_json({"type": "queue_done"})
    logger.info(f"Duplex WS connected: session={session_id} → {worker.worker_id}")
""",
        """    await ws.send_json({"type": "queue_done", "capabilities": {"binary_audio_v1": True}})
    logger.info(f"Duplex WS connected: session={session_id} → {worker.worker_id}")
""",
        '"binary_audio_v1": True',
    )

    replace_once(
        path,
        """        async def client_to_worker():
            \"\"\"Client → Worker\"\"\"
            try:
                async for raw in ws.iter_text():
                    msg = json.loads(raw)

                    if msg.get("type") == "client_diagnostic":
                        await _write_diagnostic(diag_log_path, msg)
                        continue

                    if msg.get("type") == "pause":
                        worker.update_duplex_status(GatewayWorkerStatus.DUPLEX_PAUSED)
                    elif msg.get("type") == "resume":
                        worker.update_duplex_status(GatewayWorkerStatus.DUPLEX_ACTIVE)
                    elif msg.get("type") == "stop":
                        pass

                    await worker_ws.send(raw)
            except WebSocketDisconnect:
                pass
""",
        """        async def client_to_worker():
            \"\"\"Client → Worker, accepting JSON control or binary-audio-v1 PCM.\"\"\"
            try:
                while True:
                    event = await ws.receive()
                    if event.get("type") == "websocket.disconnect":
                        break
                    raw = event.get("text")
                    if raw is None and event.get("bytes") is not None:
                        try:
                            raw = _binary_audio_to_legacy_message(event["bytes"])
                        except ValueError as error:
                            await ws.send_json({"type": "error", "error": str(error)})
                            continue
                    if raw is None:
                        continue
                    msg = json.loads(raw)

                    if msg.get("type") == "client_diagnostic":
                        await _write_diagnostic(diag_log_path, msg)
                        continue

                    if msg.get("type") == "pause":
                        worker.update_duplex_status(GatewayWorkerStatus.DUPLEX_PAUSED)
                    elif msg.get("type") == "resume":
                        worker.update_duplex_status(GatewayWorkerStatus.DUPLEX_ACTIVE)
                    elif msg.get("type") == "stop":
                        pass

                    await worker_ws.send(raw)
            except WebSocketDisconnect:
                pass
""",
        "accepting JSON control or binary-audio-v1 PCM",
    )


def patch_context_window() -> None:
    path = APP_DIR / "core/processors/unified.py"
    replace_once(
        path,
        '                "force_listen_count": self.duplex_config.force_listen_count,\n',
        """                "force_listen_count": self.duplex_config.force_listen_count,
                "sliding_window_mode": os.environ.get(
                    "MINICPM_DUPLEX_SLIDING_WINDOW_MODE", "context"
                ),
                "basic_window_high_tokens": int(os.environ.get(
                    "MINICPM_DUPLEX_WINDOW_HIGH_TOKENS", "4000"
                )),
                "basic_window_low_tokens": int(os.environ.get(
                    "MINICPM_DUPLEX_WINDOW_LOW_TOKENS", "3500"
                )),
                "context_previous_max_tokens": int(os.environ.get(
                    "MINICPM_DUPLEX_PREVIOUS_MAX_TOKENS", "500"
                )),
                "context_max_units": int(os.environ.get(
                    "MINICPM_DUPLEX_CONTEXT_MAX_UNITS", "180"
                )),
""",
        "MINICPM_DUPLEX_SLIDING_WINDOW_MODE",
    )


def patch_speaking_safety() -> None:
    path = APP_DIR / "MiniCPMO45/modeling_minicpmo_unified.py"
    replace_once(
        path,
        """        # Force listen state
        self._streaming_generate_count = 0
""",
        """        # Force listen state
        self._streaming_generate_count = 0
        self._continuous_speaking_units = 0
""",
        "self._continuous_speaking_units = 0",
    )

    replace_once(
        path,
        """        # Force listen: initial N calls OR per-chunk force_listen_override from frontend
        force_listen = self._streaming_generate_count < self.force_listen_count or force_listen_override
""",
        """        # Force listen: startup protection, client interruption, or speaking safety cap.
        max_speaking_units = int(os.environ.get("MINICPM_DUPLEX_MAX_SPEAKING_UNITS", "10"))
        safety_force_listen = self._continuous_speaking_units >= max_speaking_units
        force_listen_override = force_listen_override or safety_force_listen
        force_listen = self._streaming_generate_count < self.force_listen_count or force_listen_override
""",
        "safety_force_listen = self._continuous_speaking_units",
    )

    replace_once(
        path,
        """        if is_listen:
            self.total_hidden.append([])
            return self._make_generate_result(
""",
        """        if is_listen:
            self._continuous_speaking_units = 0
            self.total_hidden.append([])
            return self._make_generate_result(
""",
        "if is_listen:\n            self._continuous_speaking_units = 0",
    )

    replace_once(
        path,
        """        if not self.generate_audio:
            return self._make_generate_result(
""",
        """        if not self.generate_audio:
            self._continuous_speaking_units = (
                0 if end_of_turn else self._continuous_speaking_units + 1
            )
            return self._make_generate_result(
""",
        "if not self.generate_audio:\n            self._continuous_speaking_units",
    )

    replace_once(
        path,
        """        return self._make_generate_result(
            start_time, is_listen=False, text=text,
            audio_waveform=audio_waveform, end_of_turn=end_of_turn,
""",
        """        # Reliability cap: update after TTS generation.
        self._continuous_speaking_units = (
            0 if end_of_turn else self._continuous_speaking_units + 1
        )
        return self._make_generate_result(
            start_time, is_listen=False, text=text,
            audio_waveform=audio_waveform, end_of_turn=end_of_turn,
""",
        "# Reliability cap: update after TTS generation.",
    )


def main() -> None:
    patch_gateway()
    patch_context_window()
    patch_speaking_safety()


if __name__ == "__main__":
    main()
