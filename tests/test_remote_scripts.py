from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TRAINING_RUNNERS = (
    "remote_attention_clip_refinement.sh",
    "remote_shared_qkv_scale_refinement.sh",
    "remote_softmax1_refinement.sh",
    "remote_relu_hardening.sh",
    "remote_binary_qk_fallback.sh",
)
ALL_STAGES = (
    "remote_finalize_strict.sh",
    *TRAINING_RUNNERS,
    "remote_integer_rmsnorm_screen.sh",
    "remote_refinement_pipeline.sh",
)


def _read(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_every_remote_stage_has_a_unique_nonblocking_lock() -> None:
    lock_paths = []
    for name in ALL_STAGES:
        source = _read(name)
        lock_lines = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("exec 9>/tmp/")
        ]
        assert len(lock_lines) == 1, name
        assert "flock -n 9" in source
        assert "exit 75" in source
        lock_paths.append(lock_lines[0])
    assert len(lock_paths) == len(set(lock_paths))


def test_every_training_command_has_a_clean_retry_guard() -> None:
    for name in TRAINING_RUNNERS:
        source = _read(name)
        assert "source scripts/remote_stage_helpers.sh" in source
        assert source.count("uv run ternary-train") == source.count(
            "prepare_training_run"
        )


def test_downstream_stages_require_predecessor_success() -> None:
    expected_gate = {
        "remote_attention_clip_refinement.sh": '${strict_run}/SUCCESS',
        "remote_shared_qkv_scale_refinement.sh": '${clip_experiment}/SUCCESS',
        "remote_softmax1_refinement.sh": '${scale_experiment}/SUCCESS',
        "remote_relu_hardening.sh": '${normalization_experiment}/SUCCESS',
        "remote_binary_qk_fallback.sh": '${activation_experiment}/SUCCESS',
        "remote_integer_rmsnorm_screen.sh": '${binary_experiment}/SUCCESS',
    }
    for name, marker in expected_gate.items():
        source = _read(name)
        assert f'! -s "{marker}"' in source


def test_retry_helper_preserves_incomplete_runs_and_checks_exact_step() -> None:
    source = _read("remote_stage_helpers.sh")
    assert 'int(checkpoint["step"]) == int(sys.argv[2])' in source
    assert 'mv "${run_dir}" "${archived}"' in source
    assert ".incomplete-" in source


def test_strict_finalizer_requires_both_completed_artifacts() -> None:
    source = _read("remote_finalize_strict.sh")
    assert 'checkpoint-step-10000.pt \\' in source
    assert (
        '[[ -s "${run_dir}/SUCCESS" && -s "${preserved_run}/SUCCESS" ]]'
        in source
    )


def test_refinement_pipeline_waits_then_runs_declared_order() -> None:
    source = _read("remote_refinement_pipeline.sh")
    assert "while pgrep -f '^bash scripts/remote_28m_baseline" in source
    ordered = [
        "scripts/remote_finalize_strict.sh",
        "scripts/remote_attention_clip_refinement.sh",
        "scripts/remote_shared_qkv_scale_refinement.sh",
        "scripts/remote_softmax1_refinement.sh",
        "scripts/remote_relu_hardening.sh",
        "scripts/remote_binary_qk_fallback.sh",
        "scripts/remote_integer_rmsnorm_screen.sh",
        "/opt/ternary-llm-venv/bin/ternary-overnight-summary",
    ]
    offsets = [source.index(command) for command in ordered]
    assert offsets == sorted(offsets)
    assert "REFINEMENT_PIPELINE_FAILED" in source
    assert "REFINEMENT_PIPELINE_SUCCESS" in source
