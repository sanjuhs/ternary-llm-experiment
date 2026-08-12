"""Bounded native-ternary recovery pilot for MiniCPM-o 4.5.

This is deliberately a promotion-gated first stage, not a claim that an
eight-minute pilot produces a finished 9B ternary model.  It validates the
essential mechanism missing from the earlier scale-only experiment: ternary
codes may change during teacher-distilled QAT while every forward pass remains
strictly {-1, 0, +1} times a group scale.
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
)

APP_NAME = "minicpm-omni-native-ternary-qat"
GPU = "RTX-PRO-6000"
POLICY_PATH = Path("/eval/tensor-sensitivity-guardrail.json")
TRAIN_PATH = Path("/eval/cake_questions.json")
VALIDATION_PATH = Path("/eval/eval_prompts.json")
OUTPUT_DIR = Path("/native-qat/guarded-curriculum-v1")

app = modal.App(APP_NAME)
native_qat_volume = modal.Volume.from_name(
    "minicpm-omni-native-ternary-qat", create_if_missing=True
)
native_qat_image = (
    # Trackio 0.10+ requires huggingface-hub 1.x, while the audited MiniCPM
    # Transformers revision requires hub <1.0.  Keep the compatible release.
    demo_image.pip_install("trackio==0.9.1", "huggingface-hub==0.36.0")
    .add_local_file(
        local_path="src/ternary_llm/minicpmo_qat.py",
        remote_path="/app/ternary_llm/minicpmo_qat.py",
        copy=True,
    )
    .add_local_file(
        local_path="src/ternary_llm/minicpmo_native_qat.py",
        remote_path="/app/ternary_llm/minicpmo_native_qat.py",
        copy=True,
    )
    .add_local_file(
        local_path="output/minicpmo_ternary/tensor-sensitivity-guardrail.json",
        remote_path=str(POLICY_PATH),
        copy=True,
    )
    .add_local_file(
        local_path="modal_minicpm_omni/cake_questions.json",
        remote_path=str(TRAIN_PATH),
        copy=True,
    )
    .add_local_file(
        local_path="modal_minicpm_omni/eval_prompts.json",
        remote_path=str(VALIDATION_PATH),
        copy=True,
    )
)


def _selected_names(target_count: int) -> list[str]:
    policy = json.loads(POLICY_PATH.read_text())
    names = [str(name) for name in policy["selected_tensor_names"]]
    if not 1 <= target_count <= len(names):
        raise ValueError(f"target_count must be between 1 and {len(names)}")
    return names[:target_count]


def _resolve_weight_module(model: Any, tensor_name: str) -> Any:
    suffix = ".weight"
    if not tensor_name.endswith(suffix):
        raise ValueError(f"Only matrix weights are supported: {tensor_name}")
    return model.get_submodule(tensor_name[: -len(suffix)])


@app.function(
    image=native_qat_image,
    volumes={"/models": model_volume},
    cpu=2.0,
    memory=8 * 1024,
    timeout=300,
)
def plan(target_count: int = 4, steps: int = 12) -> dict[str, Any]:
    names = _selected_names(target_count)
    policy = json.loads(POLICY_PATH.read_text())
    counts = {
        str(row["tensor_name"]): int(row["parameter_count"])
        for row in policy["individual_tensor_sensitivity"]
    }
    target_parameters = sum(counts[name] for name in names)
    return {
        "format": "minicpmo-native-ternary-qat-plan-v1",
        "gpu": GPU,
        "steps": steps,
        "targets": names,
        "target_parameters": target_parameters,
        "target_fraction_of_model": target_parameters / 9_371_787_666,
        "forward_weight_alphabet": [-1, 0, 1],
        "changeable_codes": True,
        "shadow_weights_training_only": True,
        "export_shadow_weights": False,
        "promotion_gate": policy["promotion_gate"],
    }


def _evaluate(
    teacher: Any,
    student: Any,
    tokenizer: Any,
    texts: list[str],
) -> dict[str, float | int]:
    import torch
    from torch.nn import functional as F

    total_teacher_nll = 0.0
    total_student_nll = 0.0
    total_kl = 0.0
    total_tokens = 0
    top_matches = 0
    for text in texts:
        input_ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )["input_ids"].to("cuda")
        predicted = max(0, input_ids.shape[1] - 1)
        with torch.inference_mode():
            teacher_output = teacher.llm(
                input_ids=input_ids, labels=input_ids, use_cache=False
            )
            student_output = student.llm(
                input_ids=input_ids, labels=input_ids, use_cache=False
            )
            teacher_logits = teacher_output.logits[:, :-1].float()
            student_logits = student_output.logits[:, :-1].float()
            teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
            student_log_probs = torch.log_softmax(student_logits, dim=-1)
            kl = F.kl_div(
                student_log_probs,
                teacher_log_probs.exp(),
                reduction="sum",
            )
        total_teacher_nll += float(teacher_output.loss.item()) * predicted
        total_student_nll += float(student_output.loss.item()) * predicted
        total_kl += float(kl.item())
        top_matches += int(
            (teacher_logits.argmax(dim=-1) == student_logits.argmax(dim=-1))
            .sum()
            .item()
        )
        total_tokens += predicted

    teacher_loss = total_teacher_nll / total_tokens
    student_loss = total_student_nll / total_tokens
    return {
        "teacher_loss": teacher_loss,
        "student_loss": student_loss,
        "loss_delta": student_loss - teacher_loss,
        "perplexity_ratio": math.exp(min(40.0, student_loss - teacher_loss)),
        "teacher_kl_nats_per_token": total_kl / total_tokens,
        "top_token_agreement": top_matches / total_tokens,
        "tokens": total_tokens,
    }


@app.function(
    image=native_qat_image,
    gpu=GPU,
    secrets=[hf_secret],
    volumes={"/models": model_volume, "/native-qat": native_qat_volume},
    cpu=16.0,
    memory=96 * 1024,
    timeout=4 * 60 * 60,
)
def train_pilot(
    target_count: int = 4,
    steps: int = 12,
    weight_learning_rate: float = 2e-7,
    scale_learning_rate: float = 2e-5,
    temperature: float = 2.0,
    cross_entropy_weight: float = 0.05,
) -> dict[str, Any]:
    site.addsitedir("/app/.venv/base/lib/python3.10/site-packages")
    import torch
    from torch.nn import functional as F
    from transformers import AutoTokenizer

    os.chdir("/app")
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO

    from ternary_llm.minicpmo_native_qat import attach_dynamic_ternary_qat

    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    names = _selected_names(target_count)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    load_started = time.monotonic()
    load_options = {
        "trust_remote_code": True,
        "_attn_implementation": "sdpa",
        "init_vision": False,
        "init_audio": False,
        "init_tts": False,
    }
    teacher = MiniCPMO.from_pretrained(str(MODEL_DIR), **load_options).bfloat16().eval().cuda()
    student = MiniCPMO.from_pretrained(str(MODEL_DIR), **load_options).bfloat16().eval().cuda()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    parametrizations: dict[str, Any] = {}
    shadow_parameters = []
    scale_parameters = []
    initial_codes = {}
    for name in names:
        module = _resolve_weight_module(student, name)
        qat = attach_dynamic_ternary_qat(
            module,
            "weight",
            group_size=512,
            threshold=0.5,
            lloyd_iterations=8,
        )
        original = module.parametrizations.weight.original
        parametrizations[name] = (module, qat)
        shadow_parameters.append(original)
        scale_parameters.append(qat.raw_scales)
        initial_codes[name] = qat.codes(original).cpu()

    optimizer = torch.optim.AdamW(
        [
            {"params": shadow_parameters, "lr": weight_learning_rate},
            {"params": scale_parameters, "lr": scale_learning_rate},
        ],
        weight_decay=0.0,
    )
    train_texts = [str(row["question"]) for row in json.loads(TRAIN_PATH.read_text())]
    validation_texts = [
        str(row["text"]) for row in json.loads(VALIDATION_PATH.read_text())
    ]
    before = _evaluate(teacher, student, tokenizer, validation_texts)

    tracker = None
    try:
        import trackio

        tracker = trackio.init(
            project="minicpmo-native-ternary-qat",
            config={"targets": names, "steps": steps, "gpu": GPU},
        )
    except Exception as exc:
        print(f"Trackio initialization unavailable; JSON metrics remain authoritative: {exc}")

    metrics = []
    student.train()
    for step in range(steps):
        text = train_texts[step % len(train_texts)]
        input_ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )["input_ids"].cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode():
            teacher_logits = teacher.llm(input_ids=input_ids, use_cache=False).logits
        student_output = student.llm(
            input_ids=input_ids, labels=input_ids, use_cache=False
        )
        teacher_distribution = torch.softmax(
            teacher_logits.float() / temperature, dim=-1
        )
        student_log_distribution = torch.log_softmax(
            student_output.logits.float() / temperature, dim=-1
        )
        kl = F.kl_div(
            student_log_distribution,
            teacher_distribution,
            reduction="batchmean",
        ) * (temperature**2) / input_ids.shape[1]
        cross_entropy = student_output.loss.float()
        # The objective is recovery toward the frozen BF16 teacher.  A small
        # next-token term prevents pathological logit matching without pulling
        # the student materially away from the teacher distribution.
        loss = cross_entropy_weight * cross_entropy + kl
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            shadow_parameters + scale_parameters, max_norm=1.0
        )
        optimizer.step()
        row = {
            "step": step + 1,
            "loss": float(loss.item()),
            "cross_entropy": float(cross_entropy.item()),
            "distillation_kl": float(kl.item()),
            "gradient_norm": float(grad_norm.item()),
        }
        metrics.append(row)
        if tracker is not None:
            try:
                import trackio

                trackio.log(row)
            except Exception as exc:
                print(f"Trackio log warning: {exc}")

    student.eval()
    after = _evaluate(teacher, student, tokenizer, validation_texts)
    gate = {
        "absolute_loss_delta": abs(float(after["loss_delta"])),
        "absolute_loss_delta_max": 0.15,
        "perplexity_ratio": float(after["perplexity_ratio"]),
        "perplexity_ratio_min": 1 / 1.16,
        "perplexity_ratio_max": 1.16,
        "teacher_kl_nats_per_token": float(after["teacher_kl_nats_per_token"]),
        "teacher_kl_nats_per_token_max": 0.2,
        "top_token_agreement": float(after["top_token_agreement"]),
        "top_token_agreement_min": 0.8,
        "initial_absolute_loss_delta": abs(float(before["loss_delta"])),
        "initial_teacher_kl_nats_per_token": float(
            before["teacher_kl_nats_per_token"]
        ),
    }
    gate["passes_absolute_quality"] = bool(
        gate["absolute_loss_delta"] <= gate["absolute_loss_delta_max"]
        and gate["perplexity_ratio_min"]
        <= gate["perplexity_ratio"]
        <= gate["perplexity_ratio_max"]
        and gate["teacher_kl_nats_per_token"]
        <= gate["teacher_kl_nats_per_token_max"]
        and gate["top_token_agreement"] >= gate["top_token_agreement_min"]
    )
    gate["improves_over_initial"] = bool(
        gate["absolute_loss_delta"] <= gate["initial_absolute_loss_delta"]
        and gate["teacher_kl_nats_per_token"]
        <= gate["initial_teacher_kl_nats_per_token"]
    )
    gate["passes"] = bool(
        gate["passes_absolute_quality"] and gate["improves_over_initial"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {}
    changed_codes = 0
    total_codes = 0
    for name, (module, qat) in parametrizations.items():
        original = module.parametrizations.weight.original
        payload = qat.artifact_payload(original)
        changed_codes += int(
            (payload["codes"] != initial_codes[name]).sum().item()
        )
        total_codes += int(payload["codes"].numel())
        payloads[name] = payload
    checkpoint_path = OUTPUT_DIR / f"stage-{target_count:02d}-tensors.pt"
    torch.save(
        {
            "format": "minicpmo-dynamic-ternary-qat-overrides-v1",
            "base_model": "openbmb/MiniCPM-o-4_5",
            "targets": names,
            "weights": payloads,
            "shadow_weights": None,
            "quality_gate": gate,
        },
        checkpoint_path,
    )
    report = {
        "format": "minicpmo-native-ternary-qat-pilot-v1",
        "gpu": torch.cuda.get_device_name(),
        "targets": names,
        "target_parameters": sum(
            module.parametrizations.weight.original.numel()
            for module, _qat in parametrizations.values()
        ),
        "steps": steps,
        "weight_learning_rate": weight_learning_rate,
        "scale_learning_rate": scale_learning_rate,
        "temperature": temperature,
        "cross_entropy_weight": cross_entropy_weight,
        "load_ms": round((time.monotonic() - load_started) * 1000, 1),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "before": before,
        "after": after,
        "quality_gate": gate,
        "changed_code_fraction": changed_codes / total_codes,
        "checkpoint": str(checkpoint_path),
        "checkpoint_contains_shadow_weights": False,
        "metrics": metrics,
    }
    report_path = OUTPUT_DIR / f"stage-{target_count:02d}-tensors.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    native_qat_volume.commit()
    if tracker is not None:
        try:
            import trackio

            trackio.finish()
        except Exception as exc:
            print(f"Trackio finish warning: {exc}")
    return report


@app.local_entrypoint()
def main(
    action: str = "plan",
    target_count: int = 4,
    steps: int = 12,
    output: str = "",
) -> None:
    if action == "plan":
        result = plan.remote(target_count, steps)
    elif action == "pilot":
        result = train_pilot.remote(target_count, steps)
    else:
        raise ValueError("action must be one of: plan, pilot")
    rendered = json.dumps(result, indent=2) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(rendered, end="")
