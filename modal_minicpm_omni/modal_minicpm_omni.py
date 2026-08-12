"""Deploy the official MiniCPM-o 4.5 full-duplex demo on Modal.

The browser-facing gateway and the single GPU worker run in one Modal container.
Modal terminates HTTPS/WSS; the upstream gateway therefore listens on local HTTP.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

TERNARY_CANARY = os.environ.get("MINICPM_TERNARY_CANARY") == "1"
RELIABILITY_CANARY = os.environ.get("MINICPM_RELIABILITY_CANARY") == "1"
TERNARY_POLICY = (
    os.environ.get("MINICPM_TERNARY_POLICY", "super-ternary-all") if TERNARY_CANARY else ""
)
GUARDED_TERNARY = TERNARY_POLICY == "guarded-mixed-v1"
if GUARDED_TERNARY:
    APP_NAME = (
        "minicpm-omni-45-guarded-reliability-canary"
        if RELIABILITY_CANARY
        else "minicpm-omni-45-guarded-ternary"
    )
    WEB_LABEL = (
        "minicpm-omni-guarded-reliability-canary"
        if RELIABILITY_CANARY
        else "minicpm-omni-guarded-ternary"
    )
elif TERNARY_CANARY:
    APP_NAME = "minicpm-omni-45-super-ternary"
    WEB_LABEL = "minicpm-omni-super-ternary"
else:
    APP_NAME = "minicpm-omni-45-reliability-canary" if RELIABILITY_CANARY else "minicpm-omni-45"
    WEB_LABEL = "minicpm-omni-reliability-canary" if RELIABILITY_CANARY else "minicpm-omni-demo"
HF_SECRET_NAME = "huggingface-token"
MODEL_VOLUME_NAME = "minicpm-omni-cache"
TERNARY_VOLUME_NAME = "minicpm-omni-super-ternary-cache"
COMPILE_VOLUME_NAME = (
    "minicpm-omni-compile-cache-guarded-ternary"
    if GUARDED_TERNARY
    else "minicpm-omni-compile-cache-ternary-canary"
    if TERNARY_CANARY
    else "minicpm-omni-compile-cache"
)

MODEL_REPO = "openbmb/MiniCPM-o-4_5"
MODEL_DIR = Path("/models") / MODEL_REPO
DEMO_REPOSITORY = "https://github.com/OpenBMB/MiniCPM-o-Demo.git"
# Known-good direct PyTorch worker revision from 2026-05-20. Later upstream
# refactors removed the in-process chat/duplex routes before the replacement
# backend state path was complete, while this revision matches the live BF16
# control's healthy worker contract.
DEMO_REVISION = "b45ce889c5086d506434f4acc2e2b59ff7191bff"

DEMO_DIR = Path("/app")
DEMO_PYTHON = DEMO_DIR / ".venv/base/bin/python"
DEMO_PORT = 8006
WORKER_PORT = 22400
GUARDED_POLICY_PATH = Path("/app/guarded-mixed-v1-policy.json")

GPU = "L40S"
WORKER_STARTUP_TIMEOUT_SECONDS = 45 * 60
GATEWAY_STARTUP_TIMEOUT_SECONDS = 3 * 60

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
ternary_volume = modal.Volume.from_name(TERNARY_VOLUME_NAME, create_if_missing=True)
compile_volume = modal.Volume.from_name(COMPILE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)


download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


demo_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "curl",
        "ffmpeg",
        "git",
        "openssl",
        "python3-dev",
        "python3-venv",
    )
    .run_commands(
        "git init /app",
        f"git -C /app remote add origin {DEMO_REPOSITORY}",
        f"git -C /app fetch --depth 1 origin {DEMO_REVISION}",
        "git -C /app checkout --detach FETCH_HEAD",
        "cd /app && python -m venv .venv/base",
        "cd /app && .venv/base/bin/python -m pip install --upgrade pip setuptools wheel",
        "cd /app && .venv/base/bin/pip install torch==2.8.0 torchaudio==2.8.0",
        # Upstream's August requirements request librosa>=0.10.2 while the
        # published minicpmo-utils metadata still hard-pins librosa==0.9.0.
        # Install the utility package without that stale dependency constraint,
        # then let the audited demo requirements select the maintained librosa.
        "cd /app && sed -i '/^minicpmo-utils/d' requirements.txt",
        "cd /app && .venv/base/bin/pip install minicpmo-utils==1.0.6 --no-deps",
        "cd /app && .venv/base/bin/pip install -r requirements.txt",
        "cd /app && .venv/base/bin/pip install "
        "pillow==10.4.0 decord==0.6.0 moviepy==2.1.2 "
        "'onnxruntime>=1.18.0,<=1.21.0' onnx hyperpyyaml "
        "einops==0.8.1 soundfile==0.12.1 "
        "librosa==0.10.2.post1 scipy==1.15.3",
        "cd /app && cp config.example.json config.json",
        "cd /app && mkdir -p tmp data",
    )
    .add_local_file(
        local_path="modal_minicpm_omni/patches/patch_turnbased_presets.py",
        remote_path="/tmp/patch_turnbased_presets.py",
        copy=True,
    )
    .add_local_file(
        local_path="modal_minicpm_omni/patches/patch_ternary_canary.py",
        remote_path="/tmp/patch_ternary_canary.py",
        copy=True,
    )
    .add_local_file(
        local_path="modal_minicpm_omni/patches/patch_duplex_reliability.py",
        remote_path="/tmp/patch_duplex_reliability.py",
        copy=True,
    )
    .add_local_file(
        local_path="src/ternary_llm/minicpmo_canary.py",
        remote_path="/app/ternary_canary_runtime.py",
        copy=True,
    )
    .add_local_file(
        local_path="output/minicpmo_ternary/tensor-sensitivity-guardrail.json",
        remote_path=str(GUARDED_POLICY_PATH),
        copy=True,
    )
    .run_commands(
        "python /tmp/patch_turnbased_presets.py",
        "python /tmp/patch_duplex_reliability.py",
        "python /tmp/patch_ternary_canary.py",
    )
    .env(
        {
            "PYTHONPATH": "/app",
            "PYTHONUNBUFFERED": "1",
            "SKIP_MOBILE_BUILD": "1",
            "SKIP_DOCS_BUILD": "1",
            # L40S is Ada (sm89); keep its compiled artifacts isolated.
            "TORCHINDUCTOR_CACHE_DIR": "/compile-cache/torchinductor-sm89",
            "TRITON_CACHE_DIR": "/compile-cache/triton-sm89",
            "MINICPM_TERNARY_CANARY_POLICY": TERNARY_POLICY,
            "MINICPM_TERNARY_CANARY": "1" if TERNARY_CANARY else "0",
            "MINICPM_TERNARY_THRESHOLD": "0.5",
            # Preserve the latest three minutes of one-second duplex units and
            # compact older generated text into a bounded 500-token context.
            "MINICPM_DUPLEX_SLIDING_WINDOW_MODE": "context",
            "MINICPM_DUPLEX_CONTEXT_MAX_UNITS": "180",
            "MINICPM_DUPLEX_PREVIOUS_MAX_TOKENS": "500",
            "MINICPM_DUPLEX_WINDOW_HIGH_TOKENS": "4000",
            "MINICPM_DUPLEX_WINDOW_LOW_TOKENS": "3500",
            "MINICPM_DUPLEX_MAX_SPEAKING_UNITS": "10",
            "MINICPM_TERNARY_ARTIFACT_DIR": (
                "/ternary/super-ternary-all-group512-v2"
                if TERNARY_CANARY and not GUARDED_TERNARY
                else ""
            ),
            "MINICPM_TERNARY_POLICY_FILE": (
                str(GUARDED_POLICY_PATH) if GUARDED_TERNARY else ""
            ),
        }
    )
)


@app.function(
    image=download_image,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    timeout=2 * 60 * 60,
    scaledown_window=60,
)
def download_weights(revision: str | None = None) -> dict[str, str | int]:
    """Download model files once into the persistent Modal Volume."""

    from huggingface_hub import snapshot_download

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=MODEL_REPO,
        revision=revision,
        local_dir=str(MODEL_DIR),
    )
    model_volume.commit()
    total_files = sum(1 for path in Path(local_dir).rglob("*") if path.is_file())
    return {"repo": MODEL_REPO, "local_dir": local_dir, "files": total_files}


@app.function(
    image=download_image,
    volumes={"/models": model_volume},
    timeout=300,
    scaledown_window=60,
)
def cache_status() -> dict[str, object]:
    """Report whether the persistent checkpoint is ready without using a GPU."""

    total_bytes = 0
    total_files = 0
    if MODEL_DIR.exists():
        for path in MODEL_DIR.rglob("*"):
            if path.is_file():
                total_files += 1
                total_bytes += path.stat().st_size
    return {
        "model_dir": str(MODEL_DIR),
        "exists": MODEL_DIR.exists(),
        "files": total_files,
        "total_size_gb": round(total_bytes / (1024**3), 2),
    }


@app.function(
    image=demo_image,
    gpu=GPU,
    timeout=10 * 60,
    scaledown_window=60,
)
def gpu_probe() -> dict[str, object]:
    """Return the concrete GPU Modal allocated for the L40S request."""

    import site

    # The audited demo installs PyTorch inside its own virtual environment;
    # Modal invokes this lightweight probe with the image's system interpreter.
    site.addsitedir("/app/.venv/base/lib/python3.10/site-packages")
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "torch": str(torch.__version__)}
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "cuda": True,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
    }


def _write_demo_config() -> None:
    config = {
        "model": {
            "model_path": str(MODEL_DIR),
            "pt_path": None,
            # Upstream currently disables FlashAttention installation and
            # recommends PyTorch SDPA as its robust optimized path.
            "attn_implementation": "sdpa",
        },
        "audio": {
            "ref_audio_path": "assets/ref_audio/ref_minicpm_signature.wav",
            "playback_delay_ms": 120,
            "chat_vocoder": "token2wav",
        },
        "service": {
            "gateway_port": DEMO_PORT,
            "worker_base_port": WORKER_PORT,
            "max_queue_size": 8,
            "request_timeout": 600.0,
            # Eager SDPA is already real-time on L40S and starts in about a
            # minute.  The compiled path requires a source-keyed 10–16 minute
            # warmup after patches, so keep it explicit rather than the live
            # default.  Set MINICPM_ENABLE_COMPILE=1 only for dedicated tests.
            "compile": os.environ.get("MINICPM_ENABLE_COMPILE") == "1",
            "data_dir": "data",
            "eta_chat_s": 15.0,
            "eta_streaming_s": 20.0,
            "eta_audio_duplex_s": 30.0,
            "eta_omni_duplex_s": 30.0,
            "eta_ema_alpha": 0.3,
            "eta_ema_min_samples": 3,
        },
        "duplex": {"pause_timeout": 120.0},
    }
    (DEMO_DIR / "config.json").write_text(json.dumps(config, indent=2))


@app.function(
    image=demo_image,
    gpu=GPU,
    secrets=[hf_secret],
    volumes={
        "/models": model_volume,
        "/compile-cache": compile_volume,
        "/ternary": ternary_volume,
    },
    timeout=24 * 60 * 60,
    startup_timeout=50 * 60,
    scaledown_window=15 * 60,
    min_containers=0,
    max_containers=1,
    cpu=8.0,
    memory=96 * 1024,
)
@modal.web_server(DEMO_PORT, startup_timeout=50 * 60, label=WEB_LABEL)
def serve_demo() -> None:
    """Start the official worker and gateway behind Modal HTTPS/WSS."""

    import os
    import subprocess
    import threading
    import time
    import urllib.error
    import urllib.request

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"{MODEL_DIR} is missing. Run `modal run "
            "modal_minicpm_omni/modal_minicpm_omni.py --action download` first."
        )

    os.chdir(DEMO_DIR)
    _write_demo_config()

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(DEMO_DIR),
            "SKIP_MOBILE_BUILD": "1",
            "SKIP_DOCS_BUILD": "1",
        }
    )

    def stream_process(name: str, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{name}] {line}", end="", flush=True)

    def wait_for_health(
        name: str,
        url: str,
        process: subprocess.Popen[str],
        timeout_seconds: int,
        *,
        require_model_loaded: bool = False,
    ) -> dict[str, object]:
        deadline = time.time() + timeout_seconds
        last_error: object = "not started"
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited early with code {process.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not require_model_loaded or payload.get("model_loaded"):
                    print(f"[{name}] ready: {payload}", flush=True)
                    return payload
                last_error = payload
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            time.sleep(5)
        raise RuntimeError(f"{name} was not healthy after {timeout_seconds}s; last={last_error!r}")

    worker = subprocess.Popen(
        [
            str(DEMO_PYTHON),
            "worker.py",
            "--port",
            str(WORKER_PORT),
            "--gpu-id",
            "0",
            "--worker-index",
            "0",
        ],
        cwd=str(DEMO_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=stream_process, args=("worker", worker), daemon=True).start()

    wait_for_health(
        "worker",
        f"http://127.0.0.1:{WORKER_PORT}/health",
        worker,
        WORKER_STARTUP_TIMEOUT_SECONDS,
        require_model_loaded=True,
    )
    # Persist Inductor/Triton artifacts produced by the compile warmup.
    compile_volume.commit()

    gateway = subprocess.Popen(
        [
            str(DEMO_PYTHON),
            "gateway.py",
            "--port",
            str(DEMO_PORT),
            "--workers",
            f"127.0.0.1:{WORKER_PORT}",
            "--http",
        ],
        cwd=str(DEMO_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=stream_process, args=("gateway", gateway), daemon=True).start()
    wait_for_health(
        "gateway",
        f"http://127.0.0.1:{DEMO_PORT}/health",
        gateway,
        GATEWAY_STARTUP_TIMEOUT_SECONDS,
    )


@app.local_entrypoint()
def main(action: str = "cache") -> None:
    if action == "download":
        print(json.dumps(download_weights.remote(), indent=2))
    elif action == "cache":
        print(json.dumps(cache_status.remote(), indent=2))
    elif action == "gpu":
        print(json.dumps(gpu_probe.remote(), indent=2))
    else:
        raise ValueError("action must be one of: download, cache, gpu")
