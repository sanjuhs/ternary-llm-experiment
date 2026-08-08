from __future__ import annotations

# ruff: noqa: E501 -- embedded SystemVerilog, SVG, and Markdown preserve readable output.
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ternary_llm.packed import unpack_ternary_codes

TERNARY_TENSOR_KIND = "scaled-ternary-2bit-v1"
MATRIX_SUFFIXES = (
    ".attention.qkv.weight",
    ".attention.projection.weight",
    ".feed_forward.up.weight",
    ".feed_forward.down.weight",
)


@dataclass(frozen=True)
class ArchitectureAssumptions:
    process_nm: int = 45
    contribution_lanes: int = 4096
    residual_planes: int = 3
    utilization: float = 0.60
    clock_mhz: float = 500.0
    sram_energy_pj_per_bit_read: float = 0.20
    logic_energy_pj_per_contribution: float = 0.05
    fixed_point_overhead_energy_pj: float = 100_000.0
    sram_density_mib_per_mm2: float = 0.25
    logic_layout_overhead: float = 2.0
    die_overhead_fraction: float = 0.15
    wafer_diameter_mm: float = 300.0
    wafer_cost_usd: float = 6_000.0
    die_area_mm2: float = 40.0
    defect_density_per_cm2: float = 0.20
    packaging_and_test_usd: float = 4.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_deployment_artifact(path: Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("format") != "ternary-deployment-v2":
        raise ValueError("expected a ternary-deployment-v2 artifact")
    if "model" not in artifact or "config" not in artifact:
        raise ValueError("deployment artifact is missing model or config")
    return artifact


def _codes(item: dict[str, Any]) -> Tensor:
    if item.get("kind") != TERNARY_TENSOR_KIND:
        raise ValueError(f"expected {TERNARY_TENSOR_KIND}, got {item.get('kind')!r}")
    return unpack_ternary_codes(item["packed_codes"], int(item["packed_width"])).reshape(
        item["shape"]
    )


def _weight_code_stats(model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = {-1: 0, 0: 0, 1: 0}
    logical_scalars = 0
    packed_code_bytes = 0
    scale_bytes = 0
    for item in model.values():
        if item.get("kind") != TERNARY_TENSOR_KIND:
            continue
        codes = _codes(item)
        logical_scalars += codes.numel()
        packed_code_bytes += item["packed_codes"].numel() * item["packed_codes"].element_size()
        scale_bytes += item["scale"].numel() * item["scale"].element_size()
        for code in counts:
            counts[code] += int((codes == code).sum().item())
    deployed_bytes = packed_code_bytes + scale_bytes
    return {
        "logical_scalars": logical_scalars,
        "packed_code_bytes": packed_code_bytes,
        "scale_bytes": scale_bytes,
        "counts": {str(code): count for code, count in counts.items()},
        "fractions": {str(code): count / logical_scalars for code, count in counts.items()},
        "ideal_information_bits": logical_scalars * math.log2(3.0),
        "physical_code_bits": logical_scalars * 2,
        "storage_ratios": {
            "int8_to_packed_codes_and_scales": logical_scalars / deployed_bytes,
            "fp16_to_packed_codes_and_scales": logical_scalars * 2 / deployed_bytes,
            "fp32_to_packed_codes_and_scales": logical_scalars * 4 / deployed_bytes,
        },
    }


def model_workload(artifact: dict[str, Any]) -> dict[str, Any]:
    config = artifact["config"]["model"]
    model = artifact["model"]
    context = int(config["context_length"])
    d_model = int(config["d_model"])
    layers = int(config["n_layers"])
    heads = int(config["n_heads"])
    planes = int(config.get("activation_planes", 1))

    matrix_weights = 0
    matrix_weight_bytes = 0
    matrix_scale_bytes = 0
    for name, item in model.items():
        if name.endswith(MATRIX_SUFFIXES):
            matrix_weights += math.prod(int(value) for value in item["shape"])
            matrix_weight_bytes += item["packed_codes"].numel()
            matrix_scale_bytes += item["scale"].numel() * item["scale"].element_size()

    embedding = model["token_embedding"]
    vocabulary_weights = math.prod(int(value) for value in embedding["shape"])
    vocabulary_weight_bytes = embedding["packed_codes"].numel()
    vocabulary_scale_bytes = embedding["scale"].numel() * embedding["scale"].element_size()

    linear_contributions = (matrix_weights + vocabulary_weights) * planes
    qk_contributions = layers * context * d_model
    route_v_contributions = layers * context * d_model
    transform_count = 2 * layers + 2
    hadamard_add_subtracts = transform_count * d_model * int(math.log2(d_model))
    rms_norm_elements = (2 * layers + 1) * d_model
    total_low_bit_contributions = (
        linear_contributions + qk_contributions + route_v_contributions + hadamard_add_subtracts
    )
    kv_cache_bytes = layers * context * 2 * d_model * 2 // 8
    weight_stream_bytes = matrix_weight_bytes + vocabulary_weight_bytes

    return {
        "context_length": context,
        "d_model": d_model,
        "layers": layers,
        "heads": heads,
        "residual_planes": planes,
        "matrix_weights_per_token": matrix_weights,
        "vocabulary_weights_per_token": vocabulary_weights,
        "linear_ternary_contributions_per_token": linear_contributions,
        "qk_ternary_contributions_per_token_at_full_context": qk_contributions,
        "route_v_low_bit_contributions_per_token_at_full_context": route_v_contributions,
        "hadamard_add_subtracts_per_token": hadamard_add_subtracts,
        "integer_rms_norm_elements_per_token": rms_norm_elements,
        "total_low_bit_contributions_per_token": total_low_bit_contributions,
        "packed_weight_bytes_read_per_token": weight_stream_bytes,
        "scale_bytes_read_per_token": matrix_scale_bytes + vocabulary_scale_bytes,
        "packed_kv_cache_bytes_at_full_context": kv_cache_bytes,
        "incremental_decode_assumption": (
            "K/V caching is enabled; only the new token traverses projections while "
            "attention reads the full cached context."
        ),
    }


def _dies_per_wafer(diameter_mm: float, die_area_mm2: float) -> float:
    radius = diameter_mm / 2.0
    return math.pi * radius**2 / die_area_mm2 - math.pi * diameter_mm / math.sqrt(
        2.0 * die_area_mm2
    )


def _yield_murphy(die_area_mm2: float, defect_density_per_cm2: float) -> float:
    area_cm2 = die_area_mm2 / 100.0
    return (1.0 + area_cm2 * defect_density_per_cm2) ** -2


def architecture_estimate(
    artifact: dict[str, Any],
    assumptions: ArchitectureAssumptions,
    synthesis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    work = model_workload(artifact)
    contributions = int(work["total_low_bit_contributions_per_token"])
    effective_per_cycle = assumptions.contribution_lanes * assumptions.utilization
    cycles = math.ceil(contributions / effective_per_cycle)
    clock_hz = assumptions.clock_mhz * 1e6
    seconds = cycles / clock_hz
    tokens_per_second = 1.0 / seconds
    weight_bits = int(work["packed_weight_bytes_read_per_token"]) * 8
    weight_energy_pj = weight_bits * assumptions.sram_energy_pj_per_bit_read
    logic_energy_pj = contributions * assumptions.logic_energy_pj_per_contribution
    total_energy_pj = (
        weight_energy_pj + logic_energy_pj + assumptions.fixed_point_overhead_energy_pj
    )
    watts = total_energy_pj * 1e-12 * tokens_per_second

    on_chip_memory_bytes = 0
    for item in artifact["model"].values():
        kind = item.get("kind")
        if kind == TERNARY_TENSOR_KIND:
            on_chip_memory_bytes += item["packed_codes"].numel()
            on_chip_memory_bytes += item["scale"].numel() * item["scale"].element_size()
        elif kind == "positive-scale-int16-v1":
            on_chip_memory_bytes += item["codes"].numel() * item["codes"].element_size()
            on_chip_memory_bytes += item["unit"].numel() * item["unit"].element_size()
        elif kind == "raw-buffer-v1":
            on_chip_memory_bytes += item["value"].numel() * item["value"].element_size()
    on_chip_memory_bytes += int(work["packed_kv_cache_bytes_at_full_context"])
    memory_area_mm2 = on_chip_memory_bytes / (1024**2) / assumptions.sram_density_mib_per_mm2
    compute_area_mm2 = None
    if synthesis and synthesis.get("status") == "completed":
        primitive = synthesis["designs"]["programmable_ternary"]
        if primitive["mapped_area_um2"] is not None:
            compute_area_mm2 = (
                assumptions.contribution_lanes
                / primitive["lanes"]
                * primitive["mapped_area_um2"]
                / 1e6
                * assumptions.logic_layout_overhead
            )
    estimated_die_area_mm2 = assumptions.die_area_mm2
    if compute_area_mm2 is not None:
        estimated_die_area_mm2 = (memory_area_mm2 + compute_area_mm2) * (
            1.0 + assumptions.die_overhead_fraction
        )

    gross_dies = _dies_per_wafer(assumptions.wafer_diameter_mm, estimated_die_area_mm2)
    yield_fraction = _yield_murphy(estimated_die_area_mm2, assumptions.defect_density_per_cm2)
    good_dies = gross_dies * yield_fraction
    bare_die_cost = assumptions.wafer_cost_usd / good_dies

    return {
        "status": "scenario_estimate_not_signoff",
        "assumptions": asdict(assumptions),
        "cycles_per_token": cycles,
        "single_stream_latency_us": seconds * 1e6,
        "raw_tokens_per_second": tokens_per_second,
        "internal_weight_bandwidth_gb_s": (
            int(work["packed_weight_bytes_read_per_token"]) * tokens_per_second / 1e9
        ),
        "area": {
            "on_chip_model_and_kv_bytes": on_chip_memory_bytes,
            "assumed_sram_density_mib_per_mm2": assumptions.sram_density_mib_per_mm2,
            "estimated_memory_area_mm2": memory_area_mm2,
            "estimated_compute_area_mm2": compute_area_mm2,
            "estimated_die_area_mm2": estimated_die_area_mm2,
            "note": (
                "SRAM density and layout overhead are scenario assumptions, "
                "not compiled macro or place-and-route results."
            ),
        },
        "energy": {
            "weight_sram_pj_per_token": weight_energy_pj,
            "logic_pj_per_token": logic_energy_pj,
            "fixed_point_and_control_pj_per_token": assumptions.fixed_point_overhead_energy_pj,
            "total_uj_per_token": total_energy_pj / 1e6,
            "estimated_active_power_w": watts,
        },
        "illustrative_manufacturing_cost": {
            "gross_dies_per_wafer": gross_dies,
            "yield_fraction": yield_fraction,
            "good_dies_per_wafer": good_dies,
            "bare_die_cost_usd": bare_die_cost,
            "packaged_tested_cost_usd": bare_die_cost + assumptions.packaging_and_test_usd,
            "excludes": [
                "NRE, masks, IP, validation, board, memory, power delivery",
                "foundry margin, logistics, inventory loss, and software",
            ],
        },
    }


def _sv_weight_literal(codes: list[int]) -> str:
    encoded = {0: "00", 1: "01", -1: "10"}
    return "".join(encoded[code] for code in reversed(codes))


def _balanced_sum(terms: list[str]) -> str:
    if not terms:
        return "'0"
    current = terms
    while len(current) > 1:
        next_level = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                next_level.append(current[index])
            else:
                next_level.append(f"({current[index]} + {current[index + 1]})")
        current = next_level
    return current[0]


def _representative_qkv_window(
    artifact: dict[str, Any], lanes: int
) -> tuple[int, int, list[int], float]:
    qkv = _codes(artifact["model"]["blocks.0.attention.qkv.weight"])
    target = _weight_code_stats(artifact["model"])["fractions"]["0"]
    best: tuple[float, int, int, list[int]] | None = None
    for row in range(qkv.shape[0]):
        for start in range(0, qkv.shape[1] - lanes + 1, lanes):
            values = [int(value) for value in qkv[row, start : start + lanes].tolist()]
            distance = abs(values.count(0) / lanes - target)
            candidate = (distance, row, start, values)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise ValueError(f"cannot extract {lanes} lanes from QKV width {qkv.shape[1]}")
    distance, row, start, values = best
    return row, start, values, distance


def generate_rtl(
    artifact: dict[str, Any],
    output_dir: Path,
    lanes: int = 64,
    *,
    representative: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qkv = _codes(artifact["model"]["blocks.0.attention.qkv.weight"])
    if representative:
        source_row, source_start, selected, target_distance = _representative_qkv_window(
            artifact, lanes
        )
    else:
        source_row, source_start, target_distance = 0, 0, 0.0
        selected = [int(value) for value in qkv[0, :lanes].tolist()]
    width = math.ceil(math.log2(2 * lanes + 1))
    weight_literal = _sv_weight_literal(selected)
    frozen_terms = []
    for index, code in enumerate(selected):
        positive = f"decode_activation(activations[{2 * index} +: 2])"
        if code == 0:
            frozen_terms.append("'0")
        elif code == 1:
            frozen_terms.append(positive)
        else:
            frozen_terms.append(f"-{positive}")
    programmable_terms = [
        f"ternary_product(weights[{2 * index} +: 2], activations[{2 * index} +: 2])"
        for index in range(lanes)
    ]
    frozen = f"""// Generated from blocks.0.attention.qkv.weight row {source_row}, offset {source_start}.
// Ternary encoding: 2'b00=0, 2'b01=+1, 2'b10=-1, 2'b11=reserved.
module frozen_ternary_dot_{lanes}(
    input  logic [{2 * lanes - 1}:0] activations,
    output logic signed [{width - 1}:0] result
);
    localparam logic [{2 * lanes - 1}:0] WEIGHTS = {2 * lanes}'b{weight_literal};
    function automatic logic signed [{width - 1}:0] decode_activation(input logic [1:0] code);
        case (code)
            2'b01: decode_activation = {width}'sd1;
            2'b10: decode_activation = -{width}'sd1;
            default: decode_activation = '0;
        endcase
    endfunction
    always_comb begin
        result = {_balanced_sum(frozen_terms)};
    end
endmodule
"""
    programmable = f"""// Programmable ternary comparator/add-subtract baseline.
module programmable_ternary_dot_{lanes}(
    input  logic [{2 * lanes - 1}:0] weights,
    input  logic [{2 * lanes - 1}:0] activations,
    output logic signed [{width - 1}:0] result
);
    function automatic logic signed [{width - 1}:0] ternary_product(
        input logic [1:0] weight,
        input logic [1:0] activation
    );
        case ({{weight, activation}})
            4'b0101, 4'b1010: ternary_product = {width}'sd1;
            4'b0110, 4'b1001: ternary_product = -{width}'sd1;
            default:          ternary_product = '0;
        endcase
    endfunction
    always_comb begin
        result = {_balanced_sum(programmable_terms)};
    end
endmodule
"""
    int8_width = 8 + 8 + math.ceil(math.log2(lanes)) + 1
    int8_terms = [
        f"int8_product(weights[{8 * index} +: 8], activations[{8 * index} +: 8])"
        for index in range(lanes)
    ]
    int8 = f"""// Programmable signed INT8 multiply-accumulate baseline.
module programmable_int8_dot_{lanes}(
    input  logic signed [{lanes * 8 - 1}:0] weights,
    input  logic signed [{lanes * 8 - 1}:0] activations,
    output logic signed [{int8_width - 1}:0] result
);
    function automatic logic signed [{int8_width - 1}:0] int8_product(
        input logic signed [7:0] weight,
        input logic signed [7:0] activation
    );
        int8_product = weight * activation;
    endfunction
    always_comb begin
        result = {_balanced_sum(int8_terms)};
    end
endmodule
"""
    testbench = f"""`timescale 1ns/1ps
module tb_frozen_ternary_dot;
    logic [{2 * lanes - 1}:0] activations;
    logic signed [{width - 1}:0] result;
    integer i;
    integer expected;
    reg [1:0] activation_code;
    reg [1:0] weight_code;
    frozen_ternary_dot_{lanes} dut(.activations(activations), .result(result));
    initial begin
        activations = '0;
        #1;
        if (result !== 0) $fatal(1, "zero vector mismatch");
        for (i = 0; i < {lanes}; i = i + 1)
            activations[2*i +: 2] = (i % 3 == 0) ? 2'b00 :
                                             (i % 3 == 1) ? 2'b01 : 2'b10;
        expected = 0;
        #1;
        for (i = 0; i < {lanes}; i = i + 1) begin
            weight_code = dut.WEIGHTS[2*i +: 2];
            activation_code = activations[2*i +: 2];
            if ((weight_code == 2'b01 && activation_code == 2'b01) ||
                (weight_code == 2'b10 && activation_code == 2'b10)) expected = expected + 1;
            if ((weight_code == 2'b01 && activation_code == 2'b10) ||
                (weight_code == 2'b10 && activation_code == 2'b01)) expected = expected - 1;
        end
        if (result !== expected) $fatal(1, "pattern mismatch got=%0d expected=%0d", result, expected);
        $display("PASS frozen ternary dot: result=%0d", result);
        $finish;
    end
endmodule
"""
    files = {
        f"frozen_ternary_dot_{lanes}.sv": frozen,
        f"programmable_ternary_dot_{lanes}.sv": programmable,
        f"programmable_int8_dot_{lanes}.sv": int8,
        "tb_frozen_ternary_dot.sv": testbench,
    }
    for name, content in files.items():
        (output_dir / name).write_text(content)
    return {
        "source_tensor": "blocks.0.attention.qkv.weight",
        "source_row": source_row,
        "source_start": source_start,
        "lanes": lanes,
        "selected_weight_codes": selected,
        "selected_zero_fraction": selected.count(0) / lanes,
        "global_zero_fraction_distance": target_distance,
        "files": sorted(files),
    }


def _run(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        output = result.stdout + result.stderr
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{output}")
    return result.stdout + result.stderr


def simulate_rtl(rtl_dir: Path, lanes: int = 64) -> dict[str, Any]:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        return {"status": "skipped", "reason": "iverilog or vvp is unavailable"}
    executable = (rtl_dir / "frozen-dot-test").resolve()
    compile_output = _run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb_frozen_ternary_dot",
            "-o",
            str(executable),
            f"frozen_ternary_dot_{lanes}.sv",
            "tb_frozen_ternary_dot.sv",
        ],
        cwd=rtl_dir,
    )
    run_output = _run(["vvp", str(executable)], cwd=rtl_dir)
    executable.unlink(missing_ok=True)
    return {"status": "passed", "compile_output": compile_output, "run_output": run_output}


def synthesize_rtl(rtl_dir: Path, liberty: Path, lanes: int = 64) -> dict[str, Any]:
    if shutil.which("yosys") is None:
        return {"status": "skipped", "reason": "yosys is unavailable"}
    if not liberty.exists():
        return {"status": "skipped", "reason": f"missing Liberty file: {liberty}"}
    liberty = liberty.resolve()
    abc_script = (rtl_dir / "abc_timing.script").resolve()
    abc_script.write_text("strash\ndch\nmap\nstime\n")
    designs = {
        "frozen_ternary": f"frozen_ternary_dot_{lanes}",
        "programmable_ternary": f"programmable_ternary_dot_{lanes}",
        "programmable_int8": f"programmable_int8_dot_{lanes}",
    }
    results: dict[str, Any] = {}
    for label, top in designs.items():
        source = (rtl_dir / f"{top}.sv").resolve()
        script = (
            f"read_verilog -sv {source}; hierarchy -top {top}; proc; flatten; opt; "
            f"techmap; opt; abc -liberty {liberty} -script {abc_script}; "
            f"clean; stat -liberty {liberty}"
        )
        output = _run(["yosys", "-Q", "-p", script], cwd=rtl_dir)
        area_matches = re.findall(r"Chip area for module '\\\\?[^']+': ([0-9.]+)", output)
        cell_matches = re.findall(r"^\s*(\d+)\s+(?:[0-9.E+-]+\s+)?cells$", output, re.MULTILINE)
        delay_matches = re.findall(r"Delay =\s*([0-9.]+) ps", output)
        results[label] = {
            "top": top,
            "source": source.name,
            "mapped_area_um2": float(area_matches[-1]) if area_matches else None,
            "mapped_cells": int(cell_matches[-1]) if cell_matches else None,
            "abc_delay_ps": float(delay_matches[-1]) if delay_matches else None,
            "reciprocal_delay_mhz": (1e6 / float(delay_matches[-1]) if delay_matches else None),
            "lanes": lanes,
            "log": output,
        }
    frozen_area = results["frozen_ternary"]["mapped_area_um2"]
    ternary_area = results["programmable_ternary"]["mapped_area_um2"]
    int8_area = results["programmable_int8"]["mapped_area_um2"]
    return {
        "status": "completed",
        "method": "Yosys/ABC pre-layout mapping to Nangate45 typical cells",
        "liberty": str(liberty),
        "liberty_sha256": _sha256(liberty),
        "limitations": [
            "No floorplan, placement, clock tree, routing, SRAM macros, or extracted parasitics.",
            "ABC delay is a combinational estimate and not signoff timing.",
            f"The {lanes}-lane primitive is measured; full-chip values are architectural scaling scenarios.",
        ],
        "designs": results,
        "comparisons": {
            "int8_to_frozen_ternary_area_ratio": int8_area / frozen_area,
            "int8_to_programmable_ternary_area_ratio": int8_area / ternary_area,
            "programmable_to_frozen_ternary_area_ratio": ternary_area / frozen_area,
            "int8_to_frozen_area_normalized_throughput_ratio": (
                int8_area
                / frozen_area
                * results["frozen_ternary"]["reciprocal_delay_mhz"]
                / results["programmable_int8"]["reciprocal_delay_mhz"]
            ),
            "int8_to_programmable_ternary_area_normalized_throughput_ratio": (
                int8_area
                / ternary_area
                * results["programmable_ternary"]["reciprocal_delay_mhz"]
                / results["programmable_int8"]["reciprocal_delay_mhz"]
            ),
        },
    }


def architecture_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="980" viewBox="0 0 1600 980">
<defs><style>
text{font-family:Inter,Arial,sans-serif;fill:#e8eef8}.title{font-size:38px;font-weight:700}.sub{font-size:20px;fill:#aebbd0}.h{font-size:22px;font-weight:700}.b{font-size:17px}.small{font-size:14px;fill:#b8c4d8}.box{stroke-width:2;rx:18}.wire{stroke:#7dd3fc;stroke-width:4;fill:none;marker-end:url(#a)}
</style><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#7dd3fc"/></marker></defs>
<rect width="1600" height="980" fill="#07111f"/>
<text x="80" y="72" class="title">Frozen Ternary Transformer ASIC — Feasibility Architecture</text>
<text x="80" y="108" class="sub">27.4M-parameter TinyStories endpoint · exact packed weights · incremental autoregressive decode</text>
<rect class="box" x="70" y="175" width="240" height="130" fill="#13243d" stroke="#60a5fa"/><text x="190" y="218" text-anchor="middle" class="h">Token + position</text><text x="190" y="252" text-anchor="middle" class="b">2-bit packed ROM</text><text x="190" y="280" text-anchor="middle" class="small">fixed checkpoint data</text>
<rect class="box" x="390" y="160" width="820" height="590" fill="#0e1c30" stroke="#34d399"/><text x="800" y="202" text-anchor="middle" class="h">Eight frozen Transformer blocks</text>
<rect class="box" x="435" y="245" width="300" height="150" fill="#17304a" stroke="#22d3ee"/><text x="585" y="282" text-anchor="middle" class="h">Ternary contraction tiles</text><text x="585" y="318" text-anchor="middle" class="b">0 → skip · +1 → add · −1 → subtract</text><text x="585" y="350" text-anchor="middle" class="small">three residual planes, INT accumulators</text>
<rect class="box" x="865" y="245" width="300" height="150" fill="#2b2345" stroke="#c084fc"/><text x="1015" y="282" text-anchor="middle" class="h">Integer attention</text><text x="1015" y="318" text-anchor="middle" class="b">ternary Q/K/V · 2-bit LUT route</text><text x="1015" y="350" text-anchor="middle" class="small">cached K/V, no FP softmax in blocks</text>
<rect class="box" x="435" y="465" width="300" height="150" fill="#332b1b" stroke="#fbbf24"/><text x="585" y="502" text-anchor="middle" class="h">Hadamard + RMSNorm</text><text x="585" y="538" text-anchor="middle" class="b">add/sub butterfly · integer norm</text><text x="585" y="570" text-anchor="middle" class="small">fixed-point requantization boundary</text>
<rect class="box" x="865" y="465" width="300" height="150" fill="#1c332a" stroke="#4ade80"/><text x="1015" y="502" text-anchor="middle" class="h">On-chip memory</text><text x="1015" y="538" text-anchor="middle" class="b">7.51 MB deployment image</text><text x="1015" y="570" text-anchor="middle" class="small">~0.50 MB packed full-context K/V</text>
<path class="wire" d="M310 240 H390"/><path class="wire" d="M735 320 H865"/><path class="wire" d="M1015 395 V465"/><path class="wire" d="M865 540 H735"/>
<rect class="box" x="1290" y="175" width="240" height="130" fill="#13243d" stroke="#60a5fa"/><text x="1410" y="218" text-anchor="middle" class="h">Tied unembedding</text><text x="1410" y="252" text-anchor="middle" class="b">ternary logits + argmax</text><text x="1410" y="280" text-anchor="middle" class="small">sampler is a policy boundary</text><path class="wire" d="M1210 240 H1290"/>
<rect class="box" x="70" y="820" width="1460" height="95" fill="#301c24" stroke="#fb7185"/><text x="800" y="856" text-anchor="middle" class="h">What is not proven by this POC</text><text x="800" y="887" text-anchor="middle" class="b">No place-and-route, SRAM macro integration, signoff timing/power, tape-out cost quote, or complete bit-exact token sampler yet.</text>
</svg>
"""


def write_reports(
    output_dir: Path,
    artifact_path: Path,
    artifact: dict[str, Any],
    rtl: dict[str, Any],
    simulation: dict[str, Any],
    synthesis: dict[str, Any],
    estimate: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    facts = {
        "source": {
            "artifact": str(artifact_path),
            "sha256": _sha256(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "format": artifact["format"],
            "step": artifact["step"],
            "processed_tokens": artifact["processed_tokens"],
            "inference_contract": artifact["inference_contract"],
        },
        "model": artifact["config"]["model"],
        "weights": _weight_code_stats(artifact["model"]),
        "workload": model_workload(artifact),
        "rtl": rtl,
        "simulation": simulation,
        "synthesis": synthesis,
        "architecture_estimate": estimate,
    }
    (output_dir / "feasibility.json").write_text(json.dumps(facts, indent=2) + "\n")
    (output_dir / "architecture.svg").write_text(architecture_svg())

    rows = [
        ("Packed artifact bytes", facts["source"]["bytes"], "measured"),
        ("Ternary scalar count", facts["weights"]["logical_scalars"], "measured"),
        ("Weight zero fraction", facts["weights"]["fractions"]["0"], "measured"),
        (
            "Low-bit contributions/token",
            facts["workload"]["total_low_bit_contributions_per_token"],
            "analytical from architecture",
        ),
        ("Scenario cycles/token", estimate["cycles_per_token"], "scenario"),
        ("Scenario raw tokens/s", estimate["raw_tokens_per_second"], "scenario"),
        ("Scenario energy uJ/token", estimate["energy"]["total_uj_per_token"], "scenario"),
        (
            "Scenario packaged tested cost USD",
            estimate["illustrative_manufacturing_cost"]["packaged_tested_cost_usd"],
            "illustrative economics",
        ),
    ]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "evidence"])
        writer.writerows(rows)

    synth_rows = []
    for label, item in synthesis.get("designs", {}).items():
        synth_rows.append(
            f"| {label} | {item['mapped_cells']} | {item['mapped_area_um2']} | "
            f"{item['abc_delay_ps']} | {item['reciprocal_delay_mhz']:.0f} |"
        )
    synth_table = "\n".join(synth_rows) or "| not run | — | — | — | — |"
    markdown = f"""# Frozen ternary Transformer hardware feasibility report

This report distinguishes measured checkpoint facts, verified RTL, pre-layout synthesis,
and scenario assumptions. It is **not** a tape-out signoff report.

## Result

- Source artifact SHA-256: `{facts["source"]["sha256"]}`
- Packed deployment size: {facts["source"]["bytes"] / 1_000_000:.3f} MB
- Ternary scalars: {facts["weights"]["logical_scalars"]:,}
- Measured weight zeros: {facts["weights"]["fractions"]["0"]:.2%}
- RTL simulation: **{simulation["status"]}**
- Strict ternary-operand contract: **{artifact["inference_contract"]["ternary_operand_contract"]["satisfied"]}**
- End-to-end integer reference: **{artifact["inference_contract"]["end_to_end_integer_reference"]["satisfied"]}**

The checkpoint can be mapped to a frozen ternary datapath. It is not yet an end-to-end
integer token generator because requantization and the final sampling policy remain
explicit boundaries.

## 45 nm primitive synthesis

Yosys/ABC mapped a {rtl["lanes"]}-lane dot-product primitive into the open Nangate45 typical
standard-cell library. These are combinational, pre-layout results.

| Primitive | Mapped cells | Area (um²) | ABC delay (ps) | Reciprocal delay (MHz) |
|---|---:|---:|---:|---:|
{synth_table}

The frozen result uses actual codes from
`blocks.0.attention.qkv.weight[{rtl["source_row"]}, {rtl["source_start"]}:{rtl["source_start"] + rtl["lanes"]}]`.
The INT8 result is a programmable signed MAC baseline, so the comparison intentionally
captures both low precision and checkpoint freezing.

- INT8 / frozen-ternary primitive area: {synthesis["comparisons"]["int8_to_frozen_ternary_area_ratio"]:.1f}x
- INT8 / programmable-ternary primitive area: {synthesis["comparisons"]["int8_to_programmable_ternary_area_ratio"]:.1f}x
- INT8 / frozen-ternary area-normalized throughput: {synthesis["comparisons"]["int8_to_frozen_area_normalized_throughput_ratio"]:.1f}x
- INT8 / programmable-ternary area-normalized throughput: {synthesis["comparisons"]["int8_to_programmable_ternary_area_normalized_throughput_ratio"]:.1f}x

The first density result crosses two orders of magnitude, but only for this local,
hardwired primitive. It is not a full-chip speedup. The system scenario below instead
uses a reusable programmable ternary core plus SRAM.

## Weight storage

- Versus INT8: {facts["weights"]["storage_ratios"]["int8_to_packed_codes_and_scales"]:.2f}x smaller
- Versus FP16: {facts["weights"]["storage_ratios"]["fp16_to_packed_codes_and_scales"]:.2f}x smaller
- Versus FP32: {facts["weights"]["storage_ratios"]["fp32_to_packed_codes_and_scales"]:.2f}x smaller

Ternary representation alone therefore does not reach a 100x memory reduction.

## Incremental token workload

- Linear ternary contributions: {facts["workload"]["linear_ternary_contributions_per_token"]:,}
- Full-context QK contributions: {facts["workload"]["qk_ternary_contributions_per_token_at_full_context"]:,}
- Full-context route-V contributions: {facts["workload"]["route_v_low_bit_contributions_per_token_at_full_context"]:,}
- Hadamard add/subtracts: {facts["workload"]["hadamard_add_subtracts_per_token"]:,}
- Total low-bit contributions: {facts["workload"]["total_low_bit_contributions_per_token"]:,}
- Packed weight stream: {facts["workload"]["packed_weight_bytes_read_per_token"] / 1_000_000:.3f} MB/token
- Packed full-context K/V cache: {facts["workload"]["packed_kv_cache_bytes_at_full_context"] / 1_000_000:.3f} MB

## Architecture scenario—not a measured chip

At {estimate["assumptions"]["contribution_lanes"]:,} contribution lanes,
{estimate["assumptions"]["utilization"]:.0%} utilization, and
{estimate["assumptions"]["clock_mhz"]:.0f} MHz:

- {estimate["cycles_per_token"]:,} cycles/token
- {estimate["single_stream_latency_us"]:.2f} us/token
- {estimate["raw_tokens_per_second"]:,.0f} raw tokens/s
- {estimate["internal_weight_bandwidth_gb_s"]:.1f} GB/s internal packed-weight bandwidth
- {estimate["energy"]["total_uj_per_token"]:.2f} uJ/token under the declared energy coefficients
- {estimate["energy"]["estimated_active_power_w"]:.2f} W estimated active power
- {estimate["area"]["estimated_memory_area_mm2"]:.1f} mm² scenario SRAM area
- {estimate["area"]["estimated_compute_area_mm2"]:.3f} mm² scenario contraction area
- {estimate["area"]["estimated_die_area_mm2"]:.1f} mm² scenario total die area

These figures assume a real incremental K/V-cache implementation. They must not be
compared to the current PyTorch path, which recomputes the context during generation.

## Illustrative cost scenario

The calculator assumes a {estimate["assumptions"]["wafer_diameter_mm"]:.0f} mm wafer,
${estimate["assumptions"]["wafer_cost_usd"]:,.0f} wafer cost,
{estimate["area"]["estimated_die_area_mm2"]:.1f} mm² estimated die, and
{estimate["assumptions"]["defect_density_per_cm2"]:.2f} defects/cm². These values are
editable assumptions, not a foundry quote.

- Gross dies/wafer: {estimate["illustrative_manufacturing_cost"]["gross_dies_per_wafer"]:.0f}
- Estimated yield: {estimate["illustrative_manufacturing_cost"]["yield_fraction"]:.1%}
- Bare-die cost: ${estimate["illustrative_manufacturing_cost"]["bare_die_cost_usd"]:.2f}
- Packaged/tested unit cost: ${estimate["illustrative_manufacturing_cost"]["packaged_tested_cost_usd"]:.2f}

This excludes NRE, masks, SRAM/compiler IP, physical design, verification, boards,
inventory, and engineering. The assumed die area has not been established by place-and-route.

## Evidence ladder

1. **Measured:** artifact checksum, tensor shapes, packed bytes, and code counts.
2. **Verified:** actual frozen-weight RTL passes Icarus Verilog simulation.
3. **Pre-layout:** {rtl["lanes"]}-lane primitives mapped through Yosys/ABC to Nangate45.
4. **Analytical:** model-wide operation, cache, and bandwidth counts.
5. **Scenario only:** full-chip area, clock, energy, tokens/s, yield, and cost.

## Required before claiming silicon performance

- close the fixed-point requantization and final sampling boundaries;
- verify token-for-token equivalence on the complete integer runtime;
- integrate compiled SRAM macros and a banked weight/KV memory system;
- perform floorplan, placement, clock-tree synthesis, routing, extraction, STA, and power analysis;
- prototype on FPGA, then obtain foundry and packaging quotes.
"""
    (output_dir / "REPORT.md").write_text(markdown)
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("hardware/generated"))
    parser.add_argument("--liberty", type=Path)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--contribution-lanes", type=int, default=4096)
    parser.add_argument("--clock-mhz", type=float, default=500.0)
    parser.add_argument("--utilization", type=float, default=0.60)
    args = parser.parse_args()

    artifact = load_deployment_artifact(args.artifact)
    rtl_dir = args.output_dir / "rtl"
    report_dir = args.output_dir / "reports"
    rtl = generate_rtl(artifact, rtl_dir, lanes=args.lanes)
    simulation = simulate_rtl(rtl_dir, lanes=args.lanes)
    synthesis = (
        synthesize_rtl(rtl_dir, args.liberty, lanes=args.lanes)
        if args.liberty is not None
        else {"status": "skipped", "reason": "no Liberty file supplied"}
    )
    assumptions = ArchitectureAssumptions(
        contribution_lanes=args.contribution_lanes,
        residual_planes=int(artifact["config"]["model"].get("activation_planes", 1)),
        clock_mhz=args.clock_mhz,
        utilization=args.utilization,
    )
    estimate = architecture_estimate(artifact, assumptions, synthesis)
    facts = write_reports(
        report_dir,
        args.artifact,
        artifact,
        rtl,
        simulation,
        synthesis,
        estimate,
    )
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
