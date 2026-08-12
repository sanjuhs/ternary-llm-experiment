"""Audio-aware greedy guardrail for the text-safe MiniCPM-o ternary blocks."""

from __future__ import annotations

import json
import math
import site
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import modal

from modal_minicpm_omni.modal_minicpm_omni import (
    MODEL_DIR,
    demo_image,
    hf_secret,
    model_volume,
)

APP_NAME = "minicpm-omni-guarded-voice-sweep"
POLICY_PATH = Path("/eval/block-sensitivity-guardrail.json")
TENSOR_POLICY_PATH = Path("/eval/tensor-sensitivity-guardrail.json")
FIXTURES_DIR = Path("/eval/voice-fixtures")
REF_AUDIO_PATH = Path("/app/assets/ref_audio/ref_minicpm_signature.wav")
TOTAL_STORED_PARAMETERS = 9_371_787_666

voice_image = (
    demo_image.add_local_file(
        local_path="output/minicpmo_ternary/block-sensitivity-guardrail.json",
        remote_path=str(POLICY_PATH),
        copy=True,
    )
    .add_local_dir(
        local_path="output/minicpmo_ternary/voice-fixtures",
        remote_path=str(FIXTURES_DIR),
        copy=True,
    )
    .add_local_file(
        local_path="output/minicpmo_ternary/tensor-sensitivity-guardrail.json",
        remote_path=str(TENSOR_POLICY_PATH),
        copy=True,
    )
)
app = modal.App(APP_NAME)


@app.function(
    image=voice_image,
    gpu="L40S",
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    cpu=16.0,
    memory=96 * 1024,
    timeout=2 * 60 * 60,
)
def evaluate_voice_guardrail(policy_mode: str = "blocks") -> dict[str, Any]:
    site.addsitedir("/app/.venv/base/lib/python3.10/site-packages")
    import os

    import librosa
    import numpy as np
    import torch

    os.chdir("/app")
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from ternary_canary_runtime import fake_quantize_named_inplace

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if policy_mode == "blocks":
        policy_path = POLICY_PATH
        policy = json.loads(policy_path.read_text())
        selected_blocks = [int(block) for block in policy["selected_blocks"]]
        selected_names = [str(name) for name in policy["selected_tensor_names"]]
    elif policy_mode == "tensors":
        policy_path = TENSOR_POLICY_PATH
        policy = json.loads(policy_path.read_text())
        selected_names = [str(name) for name in policy["selected_tensor_names"]]
        selected_blocks = sorted(
            {int(name.split(".")[3]) for name in selected_names}
        )
    else:
        raise ValueError("policy_mode must be 'blocks' or 'tensors'")
    names_by_block = {
        block: sorted(
            name for name in selected_names if name.startswith(f"llm.model.layers.{block}.")
        )
        for block in selected_blocks
    }
    if policy_mode == "blocks" and any(
        len(names) != 7 for names in names_by_block.values()
    ):
        raise RuntimeError("Measured guarded policy no longer has seven projections per block")

    model = MiniCPMO.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, _attn_implementation="sdpa"
    ).bfloat16().eval().to("cuda")
    # Match the pinned production worker's unified-duplex initialization.  The
    # upstream revision does not expose the newer ``as_duplex`` convenience API.
    model.init_unified(
        pt_path=None,
        preload_both_tts=True,
        duplex_config={
            "generate_audio": True,
            "ls_mode": "explicit",
            "max_new_speak_tokens_per_chunk": 20,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
            "force_listen_count": 3,
        },
        device="cuda",
        chat_vocoder="token2wav",
    )
    state = model.state_dict(keep_vars=True)
    backups = {name: state[name].detach().cpu().clone() for name in selected_names}
    ref_audio, _ = librosa.load(str(REF_AUDIO_PATH), sr=16_000, mono=True)

    def restore_block(block: int) -> None:
        current = model.state_dict(keep_vars=True)
        for name in names_by_block[block]:
            current[name].data.copy_(backups[name].to(device="cuda"))

    def quantize_block(block: int) -> dict[str, Any]:
        names = names_by_block[block]
        return fake_quantize_named_inplace(
            model,
            tensor_names=names,
            threshold=0.5,
            scale_mode="lloyd-mse",
            lloyd_iterations=8,
            group_size=512,
            rows_per_chunk=8192,
            expected_tensors=7,
            expected_parameters=sum(backups[name].numel() for name in names),
        )

    def cleanup_duplex() -> None:
        if model.duplex is not None:
            model.duplex._reset_streaming_state()
            model.duplex.decoder.reset()
        if hasattr(model, "tts") and hasattr(model.tts, "audio_tokenizer"):
            tokenizer = model.tts.audio_tokenizer
            for attribute in ("stream_cache", "hift_cache_dict", "cache"):
                if hasattr(tokenizer, attribute):
                    setattr(tokenizer, attribute, None)
        model.reset_session(reset_token2wav_cache=True)
        torch.cuda.empty_cache()

    def run_fixture(fixture_id: str) -> dict[str, Any]:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        samples = np.fromfile(FIXTURES_DIR / f"{fixture_id}.f32", dtype=np.float32)
        chunks = [samples[offset : offset + 16_000] for offset in range(0, len(samples), 16_000)]
        chunks = [chunk for chunk in chunks[:8] if len(chunk) == 16_000]
        system_prompt = "You are a helpful, natural English voice assistant."
        model.duplex_prepare(
            prefix_system_prompt=f"<|im_start|>system\n{system_prompt}\n<|audio_start|>",
            suffix_system_prompt="<|audio_end|><|im_end|>",
            ref_audio=ref_audio,
            prompt_wav_path=str(REF_AUDIO_PATH),
        )
        rows = []
        generated_samples = 0
        try:
            for unit, chunk in enumerate(chunks, start=1):
                model.duplex_prefill(
                    audio_waveform=chunk,
                    frame_list=[],
                    max_slice_nums=1,
                )
                result = model.duplex_generate(
                    decode_mode="greedy",
                    temperature=0.7,
                    top_k=20,
                    top_p=0.8,
                    listen_prob_scale=1.0,
                    listen_top_k=5,
                    text_repetition_penalty=1.05,
                    text_repetition_window_size=512,
                    length_penalty=1.1,
                )
                model.duplex_finalize()
                waveform = result.get("audio_waveform")
                generated_samples += len(waveform) if waveform is not None else 0
                rows.append(
                    {
                        "unit": unit,
                        "is_listen": bool(result.get("is_listen")),
                        "text": str(result.get("text") or ""),
                        "end_of_turn": bool(result.get("end_of_turn")),
                        "audio_samples": len(waveform) if waveform is not None else 0,
                    }
                )
        finally:
            model.duplex_stop()
            cleanup_duplex()
        response_text = "".join(
            row["text"] for row in rows if row["is_listen"] is False
        ).strip()
        speaking_units = sum(row["is_listen"] is False for row in rows)
        return {
            "fixture_id": fixture_id,
            "units": rows,
            "speaking_units": speaking_units,
            "response_text": response_text,
            "generated_audio_samples": generated_samples,
            "responded": speaking_units > 0 and bool(response_text),
        }

    fixture_ids = [
        "hello",
        "gamma_bakery",
        "german_cake",
        "bangalore_weather",
        "shanghai_plan",
    ]
    bf16_panel = {fixture_id: run_fixture(fixture_id) for fixture_id in fixture_ids}
    if not all(row["responded"] for row in bf16_panel.values()):
        failed = [key for key, row in bf16_panel.items() if not row["responded"]]
        raise RuntimeError(f"BF16 control did not respond to fixtures: {failed}")
    baseline = bf16_panel["hello"]
    semantic_anchors = {
        "hello": ("doing", "well", "fine", "good", "thank", "how are"),
        "gamma_bakery": ("chocolate", "cake", "ganache", "brownie", "truffle"),
        "german_cake": ("schokol", "kuchen", "kakao", "glasur", "torte"),
        "bangalore_weather": (
            "bake",
            "bread",
            "cake",
            "cookie",
            "pastry",
            "chocolate",
        ),
        "shanghai_plan": ("serve", "bakery", "pastr", "cake", "bread", "treat"),
    }

    def compare_to_control(voice: dict[str, Any]) -> dict[str, Any]:
        control = bf16_panel[voice["fixture_id"]]
        candidate_text = voice["response_text"]
        similarity = SequenceMatcher(
            None, control["response_text"], candidate_text
        ).ratio()
        contains_cjk = any("\u3400" <= char <= "\u9fff" for char in candidate_text)
        lowered = candidate_text.casefold()
        matched_anchors = [
            anchor for anchor in semantic_anchors[voice["fixture_id"]] if anchor in lowered
        ]
        semantic_anchor_passed = bool(matched_anchors)
        passes = bool(
            voice["responded"]
            and similarity >= 0.25
            and not contains_cjk
            and semantic_anchor_passed
        )
        return {
            "fixture_id": voice["fixture_id"],
            "control_response_text": control["response_text"],
            "candidate_response_text": candidate_text,
            "subtitle_similarity_to_bf16": similarity,
            "contains_unexpected_cjk": contains_cjk,
            "matched_semantic_anchors": matched_anchors,
            "semantic_anchor_passed": semantic_anchor_passed,
            "passes": passes,
        }

    if policy_mode == "tensors":
        expected_parameters = sum(backups[name].numel() for name in selected_names)
        quantization = fake_quantize_named_inplace(
            model,
            tensor_names=selected_names,
            threshold=0.5,
            scale_mode="lloyd-mse",
            lloyd_iterations=8,
            group_size=512,
            rows_per_chunk=8192,
            expected_tensors=len(selected_names),
            expected_parameters=expected_parameters,
        )
        final_voice = [run_fixture(fixture_id) for fixture_id in fixture_ids]
        final_comparisons = [compare_to_control(voice) for voice in final_voice]
        return {
            "format": "minicpmo-projection-voice-guardrail-v1",
            "gpu": torch.cuda.get_device_name(),
            "policy_source": str(policy_path),
            "gate": {
                "must_speak": True,
                "minimum_subtitle_similarity_to_bf16": 0.25,
                "reject_unexpected_cjk_for_latin_fixtures": True,
                "require_prompt_specific_semantic_anchor": True,
                "fail_closed": True,
            },
            "bf16_voice_panel": list(bf16_panel.values()),
            "quantization": quantization,
            "selected_tensor_names": selected_names,
            "selected_tensor_count": len(selected_names),
            "selected_parameters": expected_parameters,
            "stored_parameter_coverage": expected_parameters / TOTAL_STORED_PARAMETERS,
            "final_voice_panel": final_voice,
            "final_comparisons_to_bf16": final_comparisons,
            "voice_gate_passed": all(row["passes"] for row in final_comparisons),
            "language_gate_source": str(TENSOR_POLICY_PATH),
        }

    individual = []
    for block in selected_blocks:
        quantization = quantize_block(block)
        voice = run_fixture("hello")
        restore_block(block)
        similarity = SequenceMatcher(
            None, baseline["response_text"], voice["response_text"]
        ).ratio()
        individual.append(
            {
                "block": block,
                "quantization": quantization,
                "voice": voice,
                "subtitle_similarity_to_bf16": similarity,
            }
        )

    ranking = [
        int(block)
        for block in policy["individual_ranking"]
        if int(block) in selected_blocks
    ]
    accepted: list[int] = []
    greedy_steps = []
    accepted_parameters = 0
    accepted_scales = 0
    for block in ranking:
        quantization = quantize_block(block)
        voice_panel = [run_fixture(fixture_id) for fixture_id in fixture_ids]
        comparisons = [compare_to_control(voice) for voice in voice_panel]
        passes = all(comparison["passes"] for comparison in comparisons)
        if passes:
            accepted.append(block)
            accepted_parameters += int(quantization["parameter_count"])
            accepted_scales += int(quantization["scale_count"])
            decision = "keep-ternary"
        else:
            restore_block(block)
            decision = "restore-bf16"
        greedy_steps.append(
            {
                "block": block,
                "decision": decision,
                "accepted_blocks_after_step": sorted(accepted),
                "voice_panel": voice_panel,
                "comparisons_to_bf16": comparisons,
            }
        )

    final_voice = [run_fixture(fixture_id) for fixture_id in fixture_ids]
    final_comparisons = [compare_to_control(voice) for voice in final_voice]
    estimated_bytes = (
        math.ceil(accepted_parameters / 4)
        + accepted_scales * 2
        + (TOTAL_STORED_PARAMETERS - accepted_parameters) * 2
    )
    accepted_tensor_names = sorted(
        name for block in accepted for name in names_by_block[block]
    )
    return {
        "format": "minicpmo-audio-aware-guardrail-v1",
        "gpu": torch.cuda.get_device_name(),
        "policy_source": str(POLICY_PATH),
        "gate": {
            "must_speak": True,
            "minimum_subtitle_similarity_to_bf16": 0.25,
            "reject_unexpected_cjk_for_latin_fixtures": True,
            "require_prompt_specific_semantic_anchor": True,
            "fail_closed": True,
        },
        "bf16_voice_panel": list(bf16_panel.values()),
        "individual_block_voice": individual,
        "greedy_steps": greedy_steps,
        "accepted_blocks": sorted(accepted),
        "accepted_tensor_names": accepted_tensor_names,
        "accepted_tensor_count": len(accepted_tensor_names),
        "accepted_parameters": accepted_parameters,
        "stored_parameter_coverage": accepted_parameters / TOTAL_STORED_PARAMETERS,
        "estimated_mixed_artifact_bytes": estimated_bytes,
        "estimated_mixed_artifact_gb": estimated_bytes / 1_000_000_000,
        "final_voice_panel": final_voice,
        "final_comparisons_to_bf16": final_comparisons,
        "voice_gate_passed": all(row["passes"] for row in final_comparisons),
        "requires_language_gate_recheck": True,
    }


@app.local_entrypoint()
def main(output: str = "", policy_mode: str = "blocks") -> None:
    report = evaluate_voice_guardrail.remote(policy_mode)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(rendered, end="")
