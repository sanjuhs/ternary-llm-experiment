"""Install the opt-in Stage-1 ternary hook into the pinned upstream worker."""

import os
from pathlib import Path

TARGET = Path(os.environ.get("MINICPM_TERNARY_PATCH_TARGET", "/app/core/processors/unified.py"))
NEEDLE = """        if self.device == "cuda":
            self.model.cuda()

        load_time = time.time() - start
"""
REPLACEMENT = """        if self.device == "cuda":
            self.model.cuda()

        import json

        ternary_policy = os.environ.get("MINICPM_TERNARY_CANARY_POLICY")
        if ternary_policy:
            threshold = float(os.environ.get("MINICPM_TERNARY_THRESHOLD", "0.5"))
            if ternary_policy == "llm-core":
                from ternary_canary_runtime import fake_quantize_stage1_inplace

                ternary_summary = fake_quantize_stage1_inplace(
                    self.model,
                    threshold=threshold,
                )
            elif ternary_policy == "guarded-mixed-v1":
                from ternary_canary_runtime import fake_quantize_named_inplace

                policy_path = os.environ["MINICPM_TERNARY_POLICY_FILE"]
                with open(policy_path) as policy_handle:
                    policy = json.load(policy_handle)
                tensor_names = policy["selected_tensor_names"]
                ternary_summary = fake_quantize_named_inplace(
                    self.model,
                    tensor_names=tensor_names,
                    threshold=threshold,
                    scale_mode="lloyd-mse",
                    lloyd_iterations=8,
                    group_size=512,
                    rows_per_chunk=8192,
                    expected_tensors=len(tensor_names),
                    expected_parameters=int(policy["selected_parameters"]),
                )
            elif ternary_policy == "super-ternary-all":
                artifact_dir = os.environ.get("MINICPM_TERNARY_ARTIFACT_DIR")
                if artifact_dir:
                    from ternary_canary_runtime import load_super_ternary_artifact_inplace

                    ternary_summary = load_super_ternary_artifact_inplace(
                        self.model,
                        artifact_dir=artifact_dir,
                    )
                else:
                    from ternary_canary_runtime import fake_quantize_super_ternary_inplace

                    ternary_summary = fake_quantize_super_ternary_inplace(
                        self.model,
                        checkpoint_index=os.path.join(
                            self.model_path, "model.safetensors.index.json"
                        ),
                        threshold=threshold,
                    )
            else:
                raise RuntimeError(f"Unsupported ternary canary policy: {ternary_policy}")
            logger.warning("TERNARY QUALITY CANARY ACTIVE: %s", ternary_summary)

        load_time = time.time() - start
"""


source = TARGET.read_text()
if "TERNARY QUALITY CANARY ACTIVE" in source:
    raise SystemExit(0)
if source.count(NEEDLE) != 1:
    raise RuntimeError("Pinned upstream load hook changed; refusing an unsafe patch")
TARGET.write_text(source.replace(NEEDLE, REPLACEMENT))
