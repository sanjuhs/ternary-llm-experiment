"""Build the packed all-stored-weight MiniCPM-o artifact on Modal CPU."""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "minicpm-omni-super-ternary-export"
MODEL_VOLUME_NAME = "minicpm-omni-cache"
OUTPUT_VOLUME_NAME = "minicpm-omni-super-ternary-cache"
MODEL_DIR = Path("/models/openbmb/MiniCPM-o-4_5")
ROW_OUTPUT_DIR = Path("/ternary/super-ternary-all-v1")
GROUP_OUTPUT_DIR = Path("/ternary/super-ternary-all-group512-v2")
PLAN_PATH = Path("/plans/super-ternary-all-plan.json")

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "safetensors>=0.4,<1")
    .add_local_dir(
        local_path="src/ternary_llm",
        remote_path="/pkg/ternary_llm",
        copy=True,
    )
    .add_local_file(
        local_path="output/minicpmo_ternary/super-ternary-all-plan.json",
        remote_path=str(PLAN_PATH),
        copy=True,
    )
    .env({"PYTHONPATH": "/pkg"})
)


@app.function(
    image=image,
    volumes={"/models": model_volume, "/ternary": output_volume},
    cpu=8.0,
    memory=64 * 1024,
    timeout=4 * 60 * 60,
)
def export_super_ternary(
    threshold: float = 0.5,
    variant: str = "row-v1",
) -> dict[str, object]:
    from ternary_llm.minicpmo import export_local_checkpoint

    if not MODEL_DIR.exists():
        raise RuntimeError(f"Source checkpoint is missing: {MODEL_DIR}")
    plan = json.loads(PLAN_PATH.read_text())
    if plan.get("policy") != "super-ternary-all":
        raise RuntimeError(f"Unexpected export policy: {plan.get('policy')}")
    if variant == "row-v1":
        output_dir = ROW_OUTPUT_DIR
        scale_mode = "mean-abs"
        lloyd_iterations = 4
        group_size = None
    elif variant == "group512-v2":
        output_dir = GROUP_OUTPUT_DIR
        scale_mode = "lloyd-mse"
        lloyd_iterations = 8
        group_size = 512
    else:
        raise ValueError("variant must be one of: row-v1, group512-v2")
    manifest = export_local_checkpoint(
        plan,
        checkpoint_dir=MODEL_DIR,
        output_dir=output_dir,
        threshold=threshold,
        scale_mode=scale_mode,
        lloyd_iterations=lloyd_iterations,
        group_size=group_size,
    )
    artifact_bytes = sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file())
    manifest["actual_artifact_bytes"] = artifact_bytes
    manifest["actual_artifact_gb"] = artifact_bytes / 1_000_000_000
    manifest["actual_artifact_gib"] = artifact_bytes / (1024**3)
    manifest["under_2_4_gb"] = artifact_bytes < 2_400_000_000
    manifest["variant"] = variant
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output_volume.commit()
    return manifest


@app.function(
    image=image,
    volumes={"/ternary": output_volume},
    timeout=300,
)
def artifact_status(variant: str = "row-v1") -> dict[str, object]:
    if variant == "row-v1":
        output_dir = ROW_OUTPUT_DIR
    elif variant == "group512-v2":
        output_dir = GROUP_OUTPUT_DIR
    else:
        raise ValueError("variant must be one of: row-v1, group512-v2")
    files = []
    manifest = None
    if output_dir.exists():
        files = [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        ]
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
    return {
        "output_dir": str(output_dir),
        "exists": output_dir.exists(),
        "files": files,
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "manifest": manifest,
    }


@app.local_entrypoint()
def main(action: str = "status", threshold: float = 0.5, output: str = "") -> None:
    if action == "export":
        result = export_super_ternary.remote(threshold, "row-v1")
    elif action == "export-groupwise":
        result = export_super_ternary.remote(threshold, "group512-v2")
    elif action == "status":
        result = artifact_status.remote("row-v1")
    elif action == "status-groupwise":
        result = artifact_status.remote("group512-v2")
    else:
        raise ValueError(
            "action must be one of: export, export-groupwise, status, status-groupwise"
        )
    rendered = json.dumps(result, indent=2) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(rendered, end="")
