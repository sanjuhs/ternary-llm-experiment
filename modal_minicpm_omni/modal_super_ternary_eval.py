"""Paired BF16/super-ternary language-loss benchmark on one Modal GPU."""

from __future__ import annotations

import json
import math
import os
import resource
import site
import time
from pathlib import Path
from typing import Any

import modal

from modal_minicpm_omni.modal_minicpm_omni import (
    MODEL_DIR,
    demo_image,
    hf_secret,
    model_volume,
    ternary_volume,
)

EVAL_DEVICE = os.environ.get("MINICPM_EVAL_DEVICE", "cuda")
if EVAL_DEVICE not in {"cpu", "cuda"}:
    raise ValueError("MINICPM_EVAL_DEVICE must be cpu or cuda")
APP_NAME = f"minicpm-omni-super-ternary-eval-{EVAL_DEVICE}"
GPU = "RTX-PRO-6000" if EVAL_DEVICE == "cuda" else None
PROMPTS_PATH = Path("/eval/eval_prompts.json")
TERNARY_ARTIFACT_DIR = Path("/ternary/super-ternary-all-group512-v2")

eval_image = (
    demo_image.add_local_file(
        local_path="modal_minicpm_omni/eval_prompts.json",
        remote_path=str(PROMPTS_PATH),
        copy=True,
    )
    .env({"MINICPM_EVAL_DEVICE": EVAL_DEVICE})
)
app = modal.App(APP_NAME)


def _evaluation_texts() -> list[str]:
    prompts = json.loads(PROMPTS_PATH.read_text())
    texts = [str(item["text"]) for item in prompts]
    texts.extend(
        [
            "A good chocolate cake balances bitterness, sweetness, moisture, and structure.",
            "In einer kleinen Bäckerei sind Geduld, Temperatur und gute Zutaten besonders wichtig.",
            (
                "The assistant listens to a spoken question and replies naturally, "
                "clearly, and safely."
            ),
            (
                "Opening a bakery requires a focused menu, reliable suppliers, careful "
                "costing, and kind service."
            ),
        ]
    )
    return texts


def _language_metrics(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: str,
) -> dict[str, Any]:
    import torch

    total_nll = 0.0
    total_tokens = 0
    rows = []
    started = time.monotonic()
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        labels = input_ids.clone()
        with torch.inference_mode():
            output = model.llm(input_ids=input_ids, labels=labels, use_cache=False)
        predicted_tokens = max(0, input_ids.shape[1] - 1)
        loss = float(output.loss.item())
        total_nll += loss * predicted_tokens
        total_tokens += predicted_tokens
        rows.append(
            {
                "text": text,
                "tokens": predicted_tokens,
                "loss": loss,
                "perplexity": math.exp(min(loss, 80.0)),
            }
        )
    mean_loss = total_nll / total_tokens
    return {
        "token_weighted_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "predicted_tokens": total_tokens,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
        "samples": rows,
    }


@app.function(
    image=eval_image,
    gpu=GPU,
    secrets=[hf_secret],
    volumes={"/models": model_volume, "/ternary": ternary_volume},
    cpu=16.0 if EVAL_DEVICE == "cpu" else 8.0,
    memory=96 * 1024,
    timeout=2 * 60 * 60,
)
def compare_language_quality(
    threshold: float = 0.5,
    max_texts: int = 0,
) -> dict[str, Any]:
    """Measure one loaded model before and after all-weight fake quantization."""

    site.addsitedir("/app/.venv/base/lib/python3.10/site-packages")

    import torch
    from transformers import AutoTokenizer

    os.chdir("/app")
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from ternary_canary_runtime import load_super_ternary_artifact_inplace

    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model checkpoint is missing from {MODEL_DIR}")
    if threshold != 0.5:
        raise ValueError("The persisted super-ternary artifact uses threshold 0.5")
    if EVAL_DEVICE == "cuda":
        cuda_device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        hardware_name = properties.name
        hardware_memory_gib = properties.total_memory / (1024**3)
    else:
        cuda_device = None
        hardware_name = "Modal CPU"
        hardware_memory_gib = None

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False

    load_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = MiniCPMO.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        _attn_implementation="sdpa",
    )
    model.bfloat16().eval().to(EVAL_DEVICE)
    load_ms = round((time.monotonic() - load_started) * 1000, 1)
    texts = _evaluation_texts()
    if max_texts > 0:
        texts = texts[:max_texts]

    control = _language_metrics(model, tokenizer, texts, device=EVAL_DEVICE)
    quantize_started = time.monotonic()
    quantization = load_super_ternary_artifact_inplace(
        model,
        artifact_dir=TERNARY_ARTIFACT_DIR,
    )
    quantization_ms = round((time.monotonic() - quantize_started) * 1000, 1)
    candidate = _language_metrics(model, tokenizer, texts, device=EVAL_DEVICE)

    if cuda_device is not None:
        peak_memory_gib = torch.cuda.max_memory_allocated(cuda_device) / (1024**3)
    else:
        peak_memory_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)

    return {
        "format": "minicpmo-super-ternary-language-ab-v1",
        "source_model": str(MODEL_DIR),
        "gpu": {
            "requested": GPU,
            "execution_device": EVAL_DEVICE,
            "name": hardware_name,
            "memory_gib": hardware_memory_gib,
        },
        "threshold": threshold,
        "determinism": {
            "torch_seed": 0,
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "load_ms": load_ms,
        "quantization_ms": quantization_ms,
        "peak_process_memory_gib": peak_memory_gib,
        "control_bf16": control,
        "candidate_super_ternary": candidate,
        "delta": {
            "loss": candidate["token_weighted_loss"] - control["token_weighted_loss"],
            "perplexity": candidate["perplexity"] - control["perplexity"],
            "perplexity_ratio": candidate["perplexity"] / control["perplexity"],
        },
        "quantization": quantization,
        "contract": {
            "stored_parameter_coverage": 1.0,
            "packed_artifact_executed": False,
            "packed_artifact_loaded": True,
            "runtime_storage": "BF16 values dequantized from the packed artifact",
            "warning": (
                "This benchmark loads the exact packed codes and scales, then dequantizes them "
                "for ordinary BF16 execution. It does not measure packed-kernel memory or speed."
            ),
        },
    }


@app.local_entrypoint()
def main(threshold: float = 0.5, max_texts: int = 0, output: str = "") -> None:
    report = compare_language_quality.remote(threshold, max_texts)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(rendered, end="")
