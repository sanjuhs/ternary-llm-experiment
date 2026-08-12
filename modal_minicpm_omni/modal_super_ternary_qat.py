"""Stage-1 teacher-distilled QAT for fixed ternary codes and learned group scales.

The first pilot targets the embedding and language-output boundary because the
coverage frontier proves those two tensors already cause a 4.32 loss increase.
The codes are loaded from the immutable group512-v2 artifact; only their FP32
group-scale parameters receive gradients.  Saved checkpoints contain scale
overrides and a base-artifact reference, never hidden shadow weights.
"""

from __future__ import annotations

import json
import math
import os
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

APP_NAME = "minicpm-omni-super-ternary-qat"
GPU = "RTX-PRO-6000"
ENABLE_GPU = os.environ.get("MINICPM_QAT_ENABLE_GPU") == "1"
ARTIFACT_DIR = Path("/ternary/super-ternary-all-group512-v2")
QAT_DIR = Path("/qat/group512-v2")
PROMPTS_PATH = Path("/eval/eval_prompts.json")
TARGETS = {
    "llm.model.embed_tokens.weight": "embed_tokens",
    "llm.lm_head.weight": "lm_head",
}

app = modal.App(APP_NAME)
qat_volume = modal.Volume.from_name("minicpm-omni-super-ternary-qat", create_if_missing=True)
qat_image = (
    demo_image.add_local_file(
        local_path="src/ternary_llm/minicpmo_qat.py",
        remote_path="/app/minicpmo_qat_runtime.py",
        copy=True,
    )
    .add_local_file(
        local_path="modal_minicpm_omni/eval_prompts.json",
        remote_path=str(PROMPTS_PATH),
        copy=True,
    )
)


def _artifact_manifest() -> dict[str, Any]:
    return json.loads((ARTIFACT_DIR / "manifest.json").read_text())


@app.function(
    image=qat_image,
    volumes={"/models": model_volume, "/ternary": ternary_volume},
    cpu=2.0,
    memory=8 * 1024,
    timeout=300,
)
def qat_plan() -> dict[str, Any]:
    manifest = _artifact_manifest()
    target_parameters = 2 * 621_559_808
    trainable_scales = 2 * math.ceil(621_559_808 / 512)
    return {
        "format": "minicpmo-super-ternary-qat-plan-v1",
        "stage": "language-boundary-fixed-code-scale-recovery",
        "gpu": GPU,
        "base_artifact": str(ARTIFACT_DIR),
        "base_artifact_bytes": manifest["actual_artifact_bytes"],
        "base_policy": manifest["policy"],
        "base_group_size": manifest["group_size"],
        "targets": sorted(TARGETS),
        "target_parameters": target_parameters,
        "target_parameter_fraction": target_parameters / manifest["summary"]["parameter_count"],
        "trainable_scale_parameters": trainable_scales,
        "immutable_ternary_codes": target_parameters,
        "estimated_resident_bytes": {
            "teacher_bf16_weights": manifest["summary"]["source_bytes"],
            "student_bf16_carrier_weights": manifest["summary"]["source_bytes"],
            "boundary_int8_codes_during_training": target_parameters,
            "fp32_trainable_scales": trainable_scales * 4,
            "adam_fp32_moments": trainable_scales * 8,
        },
        "checkpoint_contract": {
            "saved_values": "FP16 group-scale overrides only",
            "codes": "referenced from immutable group512-v2 artifact",
            "shadow_weights_saved": False,
            "final_parameter_alphabet": [-1, 0, 1],
            "pilot_only": True,
        },
        "promotion_gate": {
            "language_loss_delta_max": 0.15,
            "perplexity_ratio_max": 1.16,
            "required_before_next_stage": True,
        },
    }


def _load_target_items() -> dict[str, dict[str, Any]]:
    import torch

    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for name in TARGETS:
        by_shard.setdefault(index["weight_map"][name], []).append(name)
    found: dict[str, dict[str, Any]] = {}
    for source_shard, names in by_shard.items():
        path = ARTIFACT_DIR / source_shard.replace(".safetensors", ".ternary.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for name in names:
            found[name] = payload[name]
        del payload
    return found


def train_boundary_scale_pilot(
    steps: int = 8,
    learning_rate: float = 2e-3,
    temperature: float = 2.0,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    site.addsitedir("/app/.venv/base/lib/python3.10/site-packages")
    import torch
    from torch.nn import functional as F
    from torch.nn.utils import parametrize
    from transformers import AutoTokenizer

    os.chdir("/app")
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from minicpmo_qat_runtime import FixedCodeTrainableScale
    from ternary_canary_runtime import _unpack_codes

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    load_started = time.monotonic()
    teacher = MiniCPMO.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, _attn_implementation="sdpa"
    ).bfloat16().eval().to(device)
    student = MiniCPMO.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, _attn_implementation="sdpa"
    ).bfloat16().eval().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    items = _load_target_items()
    modules = {
        "llm.model.embed_tokens.weight": student.llm.model.embed_tokens,
        "llm.lm_head.weight": student.llm.lm_head,
    }
    parametrizations: dict[str, Any] = {}
    for name, module in modules.items():
        item = items[name]
        codes = _unpack_codes(item["packed_codes"], int(item["packed_width"]))
        parametrization = FixedCodeTrainableScale(
            codes=codes.to(device),
            scales=item["scale"].to(device=device, dtype=torch.float32),
            shape=tuple(item["shape"]),
        )
        parametrize.register_parametrization(module, "weight", parametrization)
        parametrizations[name] = parametrization

    trainable = [item.raw_scales for item in parametrizations.values()]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    expected_trainable = 2 * math.ceil(621_559_808 / 512)
    if trainable_count != expected_trainable:
        raise RuntimeError(
            f"trainable-scale count drifted: {trainable_count} != {expected_trainable}"
        )
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)

    texts = [str(row["text"]) for row in json.loads(PROMPTS_PATH.read_text())]
    metrics = []
    student.train()
    for step in range(steps):
        text = texts[step % len(texts)]
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = encoded["input_ids"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode():
            teacher_logits = teacher.llm(input_ids=input_ids, use_cache=False).logits
        student_output = student.llm(input_ids=input_ids, labels=input_ids, use_cache=False)
        student_logits = student_output.logits
        teacher_distribution = torch.softmax(teacher_logits.float() / temperature, dim=-1)
        student_log_distribution = torch.log_softmax(
            student_logits.float() / temperature, dim=-1
        )
        kl = F.kl_div(
            student_log_distribution,
            teacher_distribution,
            reduction="none",
        ).sum(dim=-1).mean() * (temperature**2)
        loss = student_output.loss.float() + kl
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        metrics.append(
            {
                "step": step + 1,
                "cross_entropy": float(student_output.loss.item()),
                "distillation_kl": float(kl.item()),
                "loss": float(loss.item()),
                "gradient_norm": float(gradient_norm.item()),
            }
        )

    QAT_DIR.mkdir(parents=True, exist_ok=True)
    scale_state = {
        name: parametrization.scales.detach().to(torch.float16).cpu()
        for name, parametrization in parametrizations.items()
    }
    checkpoint_path = QAT_DIR / "boundary-scale-pilot.pt"
    torch.save(
        {
            "format": "minicpmo-group512-scale-overrides-v1",
            "base_artifact": str(ARTIFACT_DIR),
            "targets": sorted(TARGETS),
            "scales": scale_state,
            "shadow_weights": None,
        },
        checkpoint_path,
    )
    report = {
        "format": "minicpmo-boundary-scale-qat-pilot-v1",
        "gpu": torch.cuda.get_device_name(),
        "steps": steps,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "trainable_scale_parameters": trainable_count,
        "immutable_code_parameters": 2 * 621_559_808,
        "load_ms": round((time.monotonic() - load_started) * 1000, 1),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "quality_gated": False,
        "pilot_only": True,
    }
    (QAT_DIR / "boundary-scale-pilot.json").write_text(json.dumps(report, indent=2) + "\n")
    qat_volume.commit()
    return report


if ENABLE_GPU:
    train_boundary_scale_pilot = app.function(
        image=qat_image,
        gpu=GPU,
        secrets=[hf_secret],
        volumes={
            "/models": model_volume,
            "/ternary": ternary_volume,
            "/qat": qat_volume,
        },
        cpu=16.0,
        memory=96 * 1024,
        timeout=4 * 60 * 60,
    )(train_boundary_scale_pilot)


@app.local_entrypoint()
def main(action: str = "plan", output: str = "", steps: int = 8) -> None:
    if action == "plan":
        result = qat_plan.remote()
    elif action == "pilot":
        if not ENABLE_GPU:
            raise RuntimeError(
                "Set MINICPM_QAT_ENABLE_GPU=1 after adding a Modal payment method"
            )
        result = train_boundary_scale_pilot.remote(steps)
    else:
        raise ValueError("action must be one of: plan, pilot")
    rendered = json.dumps(result, indent=2) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(rendered, end="")
