"""Search all-parameter ternary calibration rules on Modal CPU.

Every candidate keeps the same storage contract: one 2-bit ternary code per
stored parameter plus one FP16 scale per tensor row.  The sweep changes only
how that row scale and the zero/nonzero boundary are calibrated.
"""

from __future__ import annotations

import json
import math
import os
import re
import resource
import site
import time
from pathlib import Path
from typing import Any

import modal

from modal_minicpm_omni.modal_minicpm_omni import MODEL_DIR, demo_image, hf_secret, model_volume

APP_NAME = "minicpm-omni-super-ternary-sweep-cpu"
PROMPTS_PATH = Path("/eval/eval_prompts.json")
AUDIO_POLICY_PATH = Path("/eval/audio-aware-guardrail.json")
TOTAL_STORED_PARAMETERS = 9_371_787_666
LANGUAGE_BLOCKS = 36
LOSS_DELTA_MAX = 0.15
PERPLEXITY_RATIO_MAX = 1.16
_BLOCK_PATTERN = re.compile(r"^llm\.model\.layers\.(\d+)\.")

sweep_image = (
    demo_image.add_local_file(
        local_path="modal_minicpm_omni/eval_prompts.json",
        remote_path=str(PROMPTS_PATH),
        copy=True,
    ).add_local_file(
        local_path="output/minicpmo_ternary/audio-aware-guardrail.json",
        remote_path=str(AUDIO_POLICY_PATH),
        copy=True,
    )
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
                "The assistant listens to a spoken question and replies naturally, clearly, "
                "and safely."
            ),
            (
                "Opening a bakery requires a focused menu, reliable suppliers, careful costing, "
                "and kind service."
            ),
        ]
    )
    return texts


def _language_metrics(model: Any, tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    import torch

    total_nll = 0.0
    total_tokens = 0
    rows = []
    started = time.monotonic()
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            output = model.llm(input_ids=input_ids, labels=input_ids.clone(), use_cache=False)
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


def _metric_delta(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    loss_delta = candidate["token_weighted_loss"] - control["token_weighted_loss"]
    perplexity_ratio = candidate["perplexity"] / control["perplexity"]
    return {
        "loss": loss_delta,
        "perplexity_ratio": perplexity_ratio,
        "passes": loss_delta <= LOSS_DELTA_MAX and perplexity_ratio <= PERPLEXITY_RATIO_MAX,
    }


def _capture_teacher_distribution(
    model: Any, tokenizer: Any, texts: list[str]
) -> list[dict[str, Any]]:
    import torch

    rows = []
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            logits = model.llm(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        rows.append({"log_probs": log_probs, "top_tokens": logits.argmax(dim=-1).cpu()})
    return rows


def _alignment_metrics(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    teacher: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    total_kl = 0.0
    top_token_matches = 0
    aligned_tokens = 0
    for index, text in enumerate(texts):
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            logits = model.llm(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
        candidate_log_probs = torch.log_softmax(logits, dim=-1).cpu()
        teacher_log_probs = teacher[index]["log_probs"]
        teacher_probability = teacher_log_probs.exp()
        total_kl += float(
            (teacher_probability * (teacher_log_probs - candidate_log_probs)).sum().item()
        )
        candidate_top_tokens = logits.argmax(dim=-1).cpu()
        top_token_matches += int(
            (candidate_top_tokens == teacher[index]["top_tokens"]).sum().item()
        )
        aligned_tokens += int(candidate_top_tokens.numel())
    return {
        "teacher_kl_nats_per_token": total_kl / aligned_tokens,
        "top_token_agreement": top_token_matches / aligned_tokens,
        "aligned_tokens": aligned_tokens,
    }


def _symmetric_closeness(
    candidate: dict[str, Any],
    control: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    loss_delta = candidate["token_weighted_loss"] - control["token_weighted_loss"]
    perplexity_ratio = candidate["perplexity"] / control["perplexity"]
    passes = bool(
        abs(loss_delta) <= LOSS_DELTA_MAX
        and 1 / PERPLEXITY_RATIO_MAX <= perplexity_ratio <= PERPLEXITY_RATIO_MAX
        and alignment["teacher_kl_nats_per_token"] <= 0.2
        and alignment["top_token_agreement"] >= 0.8
    )
    return {
        "loss": loss_delta,
        "absolute_loss": abs(loss_delta),
        "perplexity_ratio": perplexity_ratio,
        "passes": passes,
    }


@app.function(
    image=sweep_image,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    cpu=16.0,
    memory=96 * 1024,
    timeout=2 * 60 * 60,
)
def evaluate_audio_aware_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Independently recheck the voice-selected allowlist on language loss."""

    site.addsitedir("/app/.venv/base/lib/python3.10/site-packages")
    import torch
    from transformers import AutoTokenizer

    os.chdir("/app")
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from ternary_canary_runtime import fake_quantize_named_inplace

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy = json.loads(AUDIO_POLICY_PATH.read_text())
    accepted_blocks = [int(block) for block in policy["accepted_blocks"]]
    requested_blocks = [int(block) for block in config.get("blocks", accepted_blocks)]
    if not set(requested_blocks).issubset(accepted_blocks):
        raise RuntimeError("Requested blocks are outside the voice-approved policy")
    tensor_names = [
        str(name)
        for name in policy["accepted_tensor_names"]
        if any(name.startswith(f"llm.model.layers.{block}.") for block in requested_blocks)
    ]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = MiniCPMO.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, _attn_implementation="sdpa"
    ).bfloat16().eval()
    texts = _evaluation_texts()
    max_texts = int(config.get("max_texts", 0))
    if max_texts > 0:
        texts = texts[:max_texts]
    control = _language_metrics(model, tokenizer, texts)
    teacher_log_probs = []
    teacher_top_tokens = []
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            logits = model.llm(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
        teacher_log_probs.append(torch.log_softmax(logits, dim=-1).cpu())
        teacher_top_tokens.append(logits.argmax(dim=-1).cpu())
    state = model.state_dict(keep_vars=True)
    expected_parameters = sum(state[name].numel() for name in tensor_names)
    quantization = fake_quantize_named_inplace(
        model,
        tensor_names=tensor_names,
        threshold=0.5,
        scale_mode="lloyd-mse",
        lloyd_iterations=8,
        group_size=512,
        rows_per_chunk=8192,
        expected_tensors=len(tensor_names),
        expected_parameters=expected_parameters,
    )
    candidate = _language_metrics(model, tokenizer, texts)
    delta = _metric_delta(candidate, control)
    total_kl = 0.0
    top_token_matches = 0
    aligned_tokens = 0
    for index, text in enumerate(texts):
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            logits = model.llm(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
        candidate_log_probs = torch.log_softmax(logits, dim=-1).cpu()
        teacher_log_prob = teacher_log_probs[index]
        teacher_probability = teacher_log_prob.exp()
        total_kl += float(
            (teacher_probability * (teacher_log_prob - candidate_log_probs)).sum().item()
        )
        candidate_top_tokens = logits.argmax(dim=-1).cpu()
        top_token_matches += int(
            (candidate_top_tokens == teacher_top_tokens[index]).sum().item()
        )
        aligned_tokens += int(candidate_top_tokens.numel())
    loss_delta = float(delta["loss"])
    perplexity_ratio = float(delta["perplexity_ratio"])
    alignment = {
        "teacher_kl_nats_per_token": total_kl / aligned_tokens,
        "top_token_agreement": top_token_matches / aligned_tokens,
        "aligned_tokens": aligned_tokens,
    }
    closeness_gate = {
        "absolute_loss_delta_max": LOSS_DELTA_MAX,
        "perplexity_ratio_min": 1 / PERPLEXITY_RATIO_MAX,
        "perplexity_ratio_max": PERPLEXITY_RATIO_MAX,
        "teacher_kl_nats_per_token_max": 0.5,
        "top_token_agreement_min": 0.7,
    }
    closeness_passed = bool(
        abs(loss_delta) <= closeness_gate["absolute_loss_delta_max"]
        and closeness_gate["perplexity_ratio_min"]
        <= perplexity_ratio
        <= closeness_gate["perplexity_ratio_max"]
        and alignment["teacher_kl_nats_per_token"]
        <= closeness_gate["teacher_kl_nats_per_token_max"]
        and alignment["top_token_agreement"]
        >= closeness_gate["top_token_agreement_min"]
    )
    return {
        "format": "minicpmo-audio-aware-language-recheck-v1",
        "policy_source": str(AUDIO_POLICY_PATH),
        "accepted_blocks": requested_blocks,
        "accepted_tensor_names": tensor_names,
        "quantization": quantization,
        "control_bf16": control,
        "candidate": candidate,
        "delta": delta,
        "alignment_to_bf16": alignment,
        "closeness_gate": closeness_gate,
        "quality_gated": closeness_passed,
    }


@app.function(
    image=sweep_image,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    cpu=16.0,
    memory=96 * 1024,
    timeout=2 * 60 * 60,
)
def evaluate_tensor_guardrail(config: dict[str, Any]) -> dict[str, Any]:
    """Search projection-level ternary coverage inside voice-safe blocks."""

    site.addsitedir("/app/.venv/base/lib/python3.10/site-packages")
    import gc

    import torch
    from transformers import AutoTokenizer

    os.chdir("/app")
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from ternary_canary_runtime import fake_quantize_named_inplace

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy = json.loads(AUDIO_POLICY_PATH.read_text())
    tensor_names = [str(name) for name in policy["accepted_tensor_names"]]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = MiniCPMO.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, _attn_implementation="sdpa"
    ).bfloat16().eval()
    texts = _evaluation_texts()
    max_texts = int(config.get("max_texts", 0))
    if max_texts > 0:
        texts = texts[:max_texts]
    control = _language_metrics(model, tokenizer, texts)
    teacher = _capture_teacher_distribution(model, tokenizer, texts)
    state = model.state_dict(keep_vars=True)

    def quantize_tensor(name: str) -> tuple[dict[str, Any], torch.Tensor]:
        backup = state[name].data.detach().clone()
        summary = fake_quantize_named_inplace(
            model,
            tensor_names=[name],
            threshold=0.5,
            scale_mode="lloyd-mse",
            lloyd_iterations=8,
            group_size=512,
            rows_per_chunk=8192,
            expected_tensors=1,
            expected_parameters=state[name].numel(),
        )
        return summary, backup

    def measure() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        candidate = _language_metrics(model, tokenizer, texts)
        alignment = _alignment_metrics(model, tokenizer, texts, teacher)
        closeness = _symmetric_closeness(candidate, control, alignment)
        return candidate, alignment, closeness

    individual = []
    for name in tensor_names:
        quantization, backup = quantize_tensor(name)
        candidate, alignment, closeness = measure()
        state[name].data.copy_(backup)
        individual.append(
            {
                "tensor_name": name,
                "parameter_count": int(state[name].numel()),
                "quantization": quantization,
                "candidate": candidate,
                "alignment_to_bf16": alignment,
                "closeness": closeness,
            }
        )
        del backup
        gc.collect()

    ranking = sorted(
        individual,
        key=lambda row: (
            not row["closeness"]["passes"],
            row["closeness"]["absolute_loss"],
            row["alignment_to_bf16"]["teacher_kl_nats_per_token"],
            row["tensor_name"],
        ),
    )
    selected_names = []
    selected_parameters = 0
    greedy_steps = []
    for sensitivity in ranking:
        name = str(sensitivity["tensor_name"])
        quantization, backup = quantize_tensor(name)
        candidate, alignment, closeness = measure()
        if closeness["passes"]:
            selected_names.append(name)
            selected_parameters += int(state[name].numel())
            decision = "keep-ternary"
        else:
            state[name].data.copy_(backup)
            decision = "restore-bf16"
        greedy_steps.append(
            {
                "tensor_name": name,
                "decision": decision,
                "selected_tensor_names_after_step": sorted(selected_names),
                "quantization": quantization,
                "candidate": candidate,
                "alignment_to_bf16": alignment,
                "closeness": closeness,
            }
        )
        del backup
        gc.collect()

    final_candidate, final_alignment, final_closeness = measure()
    return {
        "format": "minicpmo-projection-sensitivity-guardrail-v1",
        "policy_source": str(AUDIO_POLICY_PATH),
        "promotion_gate": {
            "absolute_loss_delta_max": LOSS_DELTA_MAX,
            "perplexity_ratio_min": 1 / PERPLEXITY_RATIO_MAX,
            "perplexity_ratio_max": PERPLEXITY_RATIO_MAX,
            "teacher_kl_nats_per_token_max": 0.2,
            "top_token_agreement_min": 0.8,
            "fail_closed": True,
        },
        "control_bf16": control,
        "individual_tensor_sensitivity": individual,
        "individual_ranking": [row["tensor_name"] for row in ranking],
        "greedy_steps": greedy_steps,
        "selected_tensor_names": sorted(selected_names),
        "selected_tensor_count": len(selected_names),
        "selected_parameters": selected_parameters,
        "stored_parameter_coverage": selected_parameters / TOTAL_STORED_PARAMETERS,
        "final_candidate": final_candidate,
        "final_alignment_to_bf16": final_alignment,
        "final_closeness": final_closeness,
        "quality_gated": final_closeness["passes"],
        "voice_promotion_required": True,
    }


@app.function(
    image=sweep_image,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    cpu=16.0,
    memory=96 * 1024,
    timeout=2 * 60 * 60,
)
def evaluate_block_guardrail(config: dict[str, Any]) -> dict[str, Any]:
    """Measure each language block alone, then fail-closed greedy accumulation."""

    site.addsitedir("/app/.venv/base/lib/python3.10/site-packages")
    import gc

    import torch
    from transformers import AutoTokenizer

    os.chdir("/app")
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from ternary_canary_runtime import _fake_quantize_selected_inplace, is_stage1_weight

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    load_started = time.monotonic()
    model = MiniCPMO.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, _attn_implementation="sdpa"
    ).bfloat16().eval()
    load_ms = round((time.monotonic() - load_started) * 1000, 1)
    texts = _evaluation_texts()
    max_texts = int(config.get("max_texts", 0))
    if max_texts > 0:
        texts = texts[:max_texts]
    control = _language_metrics(model, tokenizer, texts)
    state = model.state_dict(keep_vars=True)
    by_block: dict[int, list[str]] = {index: [] for index in range(LANGUAGE_BLOCKS)}
    for name in state:
        match = _BLOCK_PATTERN.match(name)
        if match and is_stage1_weight(name):
            by_block[int(match.group(1))].append(name)
    malformed = {index: names for index, names in by_block.items() if len(names) != 7}
    if malformed:
        raise RuntimeError(
            "Expected exactly seven ternary projection matrices per language block; "
            f"observed { {index: len(names) for index, names in malformed.items()} }"
        )

    quantization_kwargs = {
        "threshold": float(config.get("threshold", 0.5)),
        "scale_mode": str(config.get("scale_mode", "lloyd-mse")),
        "lloyd_iterations": int(config.get("lloyd_iterations", 8)),
        "group_size": int(config.get("group_size", 512)),
        "rows_per_chunk": int(config.get("rows_per_chunk", 8192)),
    }
    def quantize_block(block: int) -> tuple[dict[str, Any], list[torch.Tensor]]:
        selected = [(name, state[name].data) for name in sorted(by_block[block])]
        backups = [tensor.detach().clone() for _name, tensor in selected]
        summary = _fake_quantize_selected_inplace(selected, **quantization_kwargs)
        return summary, backups

    def restore_block(block: int, backups: list[torch.Tensor]) -> None:
        for name, original in zip(sorted(by_block[block]), backups, strict=True):
            state[name].data.copy_(original)

    individual = []
    for block in range(LANGUAGE_BLOCKS):
        started = time.monotonic()
        quantization, backups = quantize_block(block)
        candidate = _language_metrics(model, tokenizer, texts)
        restore_block(block, backups)
        delta = _metric_delta(candidate, control)
        individual.append(
            {
                "block": block,
                "tensor_names": sorted(by_block[block]),
                "quantization": quantization,
                "candidate": candidate,
                "delta": delta,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            }
        )
        del backups
        gc.collect()

    ranked = sorted(individual, key=lambda row: (row["delta"]["loss"], row["block"]))
    selected_blocks: list[int] = []
    greedy_steps = []
    selected_parameters = 0
    selected_scales = 0
    for sensitivity in ranked:
        block = int(sensitivity["block"])
        quantization, backups = quantize_block(block)
        candidate = _language_metrics(model, tokenizer, texts)
        delta = _metric_delta(candidate, control)
        if delta["passes"]:
            selected_blocks.append(block)
            selected_parameters += int(quantization["parameter_count"])
            selected_scales += int(quantization["scale_count"])
            decision = "keep-ternary"
        else:
            restore_block(block, backups)
            decision = "restore-bf16"
        greedy_steps.append(
            {
                "block": block,
                "decision": decision,
                "selected_blocks_after_step": sorted(selected_blocks),
                "candidate": candidate,
                "delta": delta,
            }
        )
        del backups
        gc.collect()

    final_metrics = _language_metrics(model, tokenizer, texts)
    final_delta = _metric_delta(final_metrics, control)
    estimated_bytes = (
        (selected_parameters + 3) // 4
        + selected_scales * 2
        + (TOTAL_STORED_PARAMETERS - selected_parameters) * 2
    )
    protected_contract = {
        "always_bf16_in_this_stage": [
            "llm.model.embed_tokens.weight",
            "llm.lm_head.weight",
            "normalization and bias tensors",
            "audio bridge and audio encoder",
            "speech decoder",
            "vision encoder and resampler",
        ],
        "reason": (
            "The boundary-only ablation already exceeds the global quality gate, and "
            "voice-path tensors have not yet passed isolated functional gates."
        ),
    }
    return {
        "format": "minicpmo-block-sensitivity-guardrail-v1",
        "source_model": str(MODEL_DIR),
        "execution_device": "Modal CPU",
        "load_ms": load_ms,
        "prompt_count": len(texts),
        "promotion_gate": {
            "loss_delta_max": LOSS_DELTA_MAX,
            "perplexity_ratio_max": PERPLEXITY_RATIO_MAX,
            "fail_closed": True,
        },
        "quantization": quantization_kwargs,
        "control_bf16": control,
        "individual_block_sensitivity": individual,
        "individual_ranking": [int(row["block"]) for row in ranked],
        "greedy_steps": greedy_steps,
        "selected_blocks": sorted(selected_blocks),
        "selected_tensor_names": sorted(
            name for block in selected_blocks for name in by_block[block]
        ),
        "selected_parameters": selected_parameters,
        "stored_parameter_coverage": selected_parameters / TOTAL_STORED_PARAMETERS,
        "final_candidate": final_metrics,
        "final_delta": final_delta,
        "estimated_mixed_artifact_bytes": estimated_bytes,
        "estimated_mixed_artifact_gb": estimated_bytes / 1_000_000_000,
        "protected_contract": protected_contract,
        "voice_promotion_required": True,
        "quality_gated": final_delta["passes"],
    }


@app.function(
    image=sweep_image,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    cpu=16.0,
    memory=96 * 1024,
    timeout=2 * 60 * 60,
)
def evaluate_configuration(config: dict[str, Any]) -> dict[str, Any]:
    site.addsitedir("/app/.venv/base/lib/python3.10/site-packages")

    from transformers import AutoTokenizer

    os.chdir("/app")
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from ternary_canary_runtime import (
        _fake_quantize_selected_inplace,
        fake_quantize_stage1_inplace,
        fake_quantize_super_ternary_inplace,
    )

    threshold = float(config["threshold"])
    scale_mode = str(config["scale_mode"])
    lloyd_iterations = int(config.get("lloyd_iterations", 4))
    configured_group_size = int(config.get("group_size", 0))
    group_size = configured_group_size or None
    rows_per_chunk = int(config.get("rows_per_chunk", 8_192 if group_size else 256))
    max_texts = int(config.get("max_texts", 0))
    selection = str(config.get("selection", "all"))

    import torch

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    load_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = MiniCPMO.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        _attn_implementation="sdpa",
    ).bfloat16().eval()
    load_ms = round((time.monotonic() - load_started) * 1000, 1)
    texts = _evaluation_texts()
    if max_texts > 0:
        texts = texts[:max_texts]

    control = _language_metrics(model, tokenizer, texts)
    quantize_started = time.monotonic()
    quantization_kwargs = {
        "threshold": threshold,
        "scale_mode": scale_mode,
        "lloyd_iterations": lloyd_iterations,
        "group_size": group_size,
        "rows_per_chunk": rows_per_chunk,
    }
    if selection == "all":
        quantization = fake_quantize_super_ternary_inplace(
            model,
            checkpoint_index=MODEL_DIR / "model.safetensors.index.json",
            **quantization_kwargs,
        )
    elif selection == "llm-core":
        quantization = fake_quantize_stage1_inplace(model, **quantization_kwargs)
    elif selection in {"llm-all", "llm-boundary"}:
        index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
        if selection == "llm-all":
            target_names = [name for name in index["weight_map"] if name.startswith("llm.")]
        else:
            target_names = ["llm.lm_head.weight", "llm.model.embed_tokens.weight"]
        state = model.state_dict(keep_vars=True)
        selected = [(name, state[name].data) for name in target_names]
        quantization = _fake_quantize_selected_inplace(selected, **quantization_kwargs)
        quantization.update(
            {
                "contract": f"{selection}-diagnostic-fake-quantization",
                "all_stored_parameters_ternary": False,
                "packed_storage": False,
                "custom_ternary_kernels": False,
                "quality_canary_only": True,
            }
        )
    else:
        raise ValueError("selection must be one of: all, llm-core, llm-all, llm-boundary")
    quantization_ms = round((time.monotonic() - quantize_started) * 1000, 1)
    candidate = _language_metrics(model, tokenizer, texts)
    selected_parameters = int(quantization["parameter_count"])
    estimated_packed_bytes = (
        (selected_parameters + 3) // 4
        + quantization["scale_count"] * 2
        + (TOTAL_STORED_PARAMETERS - selected_parameters) * 2
    )
    return {
        "selection": selection,
        "threshold": threshold,
        "scale_mode": scale_mode,
        "lloyd_iterations": lloyd_iterations if scale_mode == "lloyd-mse" else 0,
        "group_size": group_size,
        "rows_per_chunk": rows_per_chunk,
        "load_ms": load_ms,
        "quantization_ms": quantization_ms,
        "peak_process_memory_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2),
        "control_bf16": control,
        "candidate_super_ternary": candidate,
        "delta": {
            "loss": candidate["token_weighted_loss"] - control["token_weighted_loss"],
            "perplexity_ratio": candidate["perplexity"] / control["perplexity"],
        },
        "quantization": quantization,
        "estimated_packed_bytes": estimated_packed_bytes,
        "estimated_packed_gb": estimated_packed_bytes / 1_000_000_000,
        "estimated_under_2_4_gb": estimated_packed_bytes < 2_400_000_000,
        "contract": {
            "all_stored_parameters_ternary": selection == "all",
            "stored_parameter_coverage": selected_parameters / TOTAL_STORED_PARAMETERS,
            "packed_storage_estimate_unchanged": True,
            "packed_artifact_executed": False,
            "runtime_storage": "BF16 fake-quantized values",
        },
    }


@app.local_entrypoint()
def main(output: str = "", max_texts: int = 0, campaign: str = "groupwise") -> None:
    if campaign == "groupwise":
        configurations = [
            {"threshold": 0.5, "scale_mode": "mean-abs", "group_size": 512},
            {
                "threshold": 0.5,
                "scale_mode": "lloyd-mse",
                "lloyd_iterations": 8,
                "group_size": 512,
            },
            {
                "threshold": 0.5,
                "scale_mode": "lloyd-mse",
                "lloyd_iterations": 8,
                "group_size": 1024,
            },
            {
                "threshold": 0.35,
                "scale_mode": "lloyd-mse",
                "lloyd_iterations": 8,
                "group_size": 512,
            },
        ]
        report_format = "minicpmo-super-ternary-groupwise-sweep-v1"
    elif campaign == "frontier":
        configurations = [
            {
                "selection": selection,
                "threshold": 0.5,
                "scale_mode": "lloyd-mse",
                "lloyd_iterations": 8,
                "group_size": 512,
            }
            for selection in ("llm-boundary", "llm-core", "llm-all", "all")
        ]
        report_format = "minicpmo-ternary-coverage-frontier-v1"
    elif campaign == "blocks":
        report = evaluate_block_guardrail.remote(
            {
                "threshold": 0.5,
                "scale_mode": "lloyd-mse",
                "lloyd_iterations": 8,
                "group_size": 512,
                "max_texts": max_texts,
            }
        )
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
        print(rendered, end="")
        return
    elif campaign == "audio-policy":
        report = evaluate_audio_aware_policy.remote({"max_texts": max_texts})
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
        print(rendered, end="")
        return
    elif campaign == "audio-frontier":
        configurations = [
            {"blocks": blocks, "max_texts": max_texts}
            for blocks in ([5], [5, 23], [5, 23, 19])
        ]
        rows = list(evaluate_audio_aware_policy.map(configurations, return_exceptions=False))
        report = {
            "format": "minicpmo-audio-aware-language-frontier-v1",
            "configurations": rows,
        }
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
        print(rendered, end="")
        return
    elif campaign == "tensors":
        report = evaluate_tensor_guardrail.remote({"max_texts": max_texts})
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
        print(rendered, end="")
        return
    else:
        raise ValueError(
            "campaign must be one of: groupwise, frontier, blocks, audio-policy, "
            "audio-frontier, tensors"
        )
    for config in configurations:
        config["max_texts"] = max_texts
    rows = list(evaluate_configuration.map(configurations, return_exceptions=False))
    rows.sort(key=lambda row: row["candidate_super_ternary"]["token_weighted_loss"])
    report = {
        "format": report_format,
        "source_model": str(MODEL_DIR),
        "execution_device": "Modal CPU",
        "configuration_count": len(rows),
        "best": rows[0],
        "configurations": rows,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(rendered, end="")
