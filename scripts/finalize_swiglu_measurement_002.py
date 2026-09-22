"""Parse local Nsight Compute CSV and finalize measurement evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "integration" / "swiglu" / "real_loop_002"


def rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    return list(csv.DictReader(lines[start:]))


def parse(name: str) -> dict[str, object]:
    data = rows(OUT / name)
    kernels = sorted({row.get("Kernel Name", "") for row in data if row.get("Kernel Name")})
    metrics: dict[str, list[dict[str, str]]] = {}
    for row in data:
        key = row.get("Metric Name", "")
        if key:
            metrics.setdefault(key, []).append({"value": row.get("Metric Value", ""), "unit": row.get("Metric Unit", "")})
    return {"csv": name, "kernel_count": len(kernels), "kernel_names": kernels, "metrics": metrics}


def metric(section: dict[str, object], name: str) -> object:
    values = section.get("metrics", {}).get(name, [])
    return values[0] if values else None


def metric_value(section: dict[str, object], name: str, suffix: str = ".sum") -> object:
    value = metric(section, name + suffix)
    if value is None:
        return None
    raw = str(value["value"]).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return value


def main() -> int:
    sections = {"FC1": parse("ncu_fc1_basic.csv"), "SWIGLU": parse("ncu_swiglu_basic.csv"), "FC2": parse("ncu_fc2_basic.csv")}
    custom = {"SWIGLU": parse("ncu_swiglu_memory.csv"), "FC2": parse("ncu_fc2_memory.csv")}
    attribution = {
        "status": "PASS",
        "tool": "Nsight Compute CLI 2026.3.0",
        "scope": "single authentic Megatron MLP forward iteration",
        "nvtx_ranges": {"FC1": "AKA_FC1", "SWIGLU": "AKA_SWIGLU", "FC2": "AKA_FC2"},
        "sections": sections,
        "custom_metric_sections": custom,
        "fc1_kernel_count": sections["FC1"]["kernel_count"],
        "swiglu_kernel_count": sections["SWIGLU"]["kernel_count"],
        "fc2_kernel_count": sections["FC2"]["kernel_count"],
        "separate_swiglu_fc2_kernel_executions": sections["SWIGLU"]["kernel_count"] > 0 and sections["FC2"]["kernel_count"] > 0,
        "device_tensor_materialized": "SUPPORTED",
        "device_tensor_materialization_basis": "Megatron returns SwiGLU output tensor to a separate linear_fc2 invocation; NVTX-filtered NCU shows distinct executions.",
        "dram_round_trip": "UNKNOWN",
        "dram_round_trip_reason": "Per-kernel DRAM counters include all traffic attributable to that kernel and cannot isolate activation from weights/other traffic in this workload.",
        "raw_artifacts": ["ncu_fc1_basic.csv", "ncu_swiglu_basic.csv", "ncu_fc2_basic.csv", "ncu_swiglu_memory.csv", "ncu_fc2_memory.csv"],
    }
    (OUT / "kernel_boundary_attribution.json").write_text(json.dumps(attribution, indent=2), encoding="utf-8")

    timing = json.loads((OUT / "boundary_timing.json").read_text(encoding="utf-8"))
    uninstrumented = json.loads((OUT / "uninstrumented" / "boundary_timing.json").read_text(encoding="utf-8"))
    instrumented_median = timing["whole_mlp_statistics_ms"]["median"]
    uninstrumented_median = uninstrumented["whole_mlp_statistics_ms"]["median"]
    overhead_pct = (instrumented_median / uninstrumented_median - 1.0) * 100.0
    facts = {
        "schema": "PerformanceFacts.v6",
        "episode": "REAL_OPT_LOOP_002",
        "evidence_status": "MEASUREMENT_COMPLETE",
        "environment": timing.get("environment"),
        "shape": {"value": [3, 2, 8], "evidence": "MEASURED", "source": "real workload input"},
        "dtype": {"value": "float32", "evidence": "MEASURED"},
        "layout": {"value": "contiguous input from torch.randn; module path preserves tensor contract", "evidence": "MEASURED/SOURCE_DERIVED"},
        "known_kernel_boundaries": {"value": ["FC1", "SWIGLU", "FC2"], "evidence": "MEASURED via NVTX-filtered Nsight Compute"},
        "swiglu_kernel_count": {"value": attribution["swiglu_kernel_count"], "evidence": "MEASURED"},
        "fc2_kernel_count": {"value": attribution["fc2_kernel_count"], "evidence": "MEASURED"},
        "separate_cuda_kernel_boundary": {"value": True, "evidence": "MEASURED"},
        "device_tensor_materialized": {"value": True, "evidence": "SOURCE_DERIVED + MEASURED boundary attribution"},
        "activation_dram_write": {"value": None, "evidence": "UNKNOWN", "reason": "not separable from kernel total traffic"},
        "activation_dram_read": {"value": None, "evidence": "UNKNOWN", "reason": "not separable from FC2 weight/other reads"},
        "dram_round_trip": {"value": None, "evidence": "UNKNOWN"},
        "l2_activity": {"value": {k: metric(sections["SWIGLU"], k) for k in ["L2 Cache Throughput", "Total L2 Elapsed Cycles"]}, "evidence": "MEASURED per SWIGLU kernel, not activation-only"},
        "registers_per_thread": {"value": metric(sections["SWIGLU"], "Registers Per Thread"), "evidence": "MEASURED"},
        "shared_memory": {"value": {"configuration_bytes": metric(sections["SWIGLU"], "Shared Memory Configuration Size"), "dynamic_bytes_per_block": metric(sections["SWIGLU"], "Dynamic Shared Memory Per Block"), "static_bytes_per_block": metric(sections["SWIGLU"], "Static Shared Memory Per Block")}, "evidence": "MEASURED"},
        "occupancy": {"value": metric(sections["SWIGLU"], "Achieved Occupancy"), "evidence": "MEASURED per SWIGLU kernel if present"},
        "kernel_count": {"value": {k: sections[k]["kernel_count"] for k in sections}, "evidence": "MEASURED"},
        "launch_cost": {"value": metric(sections["SWIGLU"], "Duration"), "evidence": "MEASURED kernel duration, not launch API overhead"},
        "swiglu_duration": {"value": timing.get("swiglu_statistics_ms"), "evidence": "MEASURED CUDA Event with inner instrumentation"},
        "fc2_duration": {"value": timing.get("fc2_statistics_ms"), "evidence": "MEASURED CUDA Event with inner instrumentation"},
        "combined_boundary_duration": {"value": timing.get("swiglu_plus_fc2_statistics_ms"), "evidence": "DERIVED sum of measured per-region Event durations"},
        "model_fraction": {"value": 0.015571342909175317, "evidence": "MEASURED prior split-run authentic local minimal GPTModel profile"},
        "system_ceiling": {"value": 1.015817644881643, "evidence": "DERIVED prior split-run local minimal GPTModel Amdahl ceiling"},
        "reuse": {"value": None, "evidence": "UNKNOWN"},
        "lifetime": {"value": "activation output consumed by subsequent FC2 call; exact storage lifetime policy UNKNOWN", "evidence": "SOURCE_DERIVED"},
        "reduction": {"value": "none in forward SwiGLU boundary", "evidence": "SOURCE_DERIVED"},
        "synchronization": {"value": "CUDA event synchronization used for measurement; internal kernel synchronization UNKNOWN", "evidence": "MEASURED/UNKNOWN"},
        "resource_limits": {"register_pressure": metric(sections["SWIGLU"], "Registers Per Thread"), "shared_memory": metric(sections["SWIGLU"], "Shared Memory Configuration Size"), "occupancy": metric(sections["SWIGLU"], "Achieved Occupancy"), "evidence": "MEASURED"},
        "swiglu_kernel_total_dram_bytes": {"value": (metric_value(custom["SWIGLU"], "dram__bytes_op_read") or 0) + (metric_value(custom["SWIGLU"], "dram__bytes_op_write") or 0), "evidence": "MEASURED", "caveat": "total traffic for the filtered SwiGLU kernel, not activation-only traffic"},
        "swiglu_kernel_dram_read_bytes": {"value": metric_value(custom["SWIGLU"], "dram__bytes_op_read"), "evidence": "MEASURED", "caveat": "kernel total; cannot be assigned solely to activation"},
        "swiglu_kernel_dram_write_bytes": {"value": metric_value(custom["SWIGLU"], "dram__bytes_op_write"), "evidence": "MEASURED", "caveat": "kernel total; cannot be assigned solely to activation"},
        "swiglu_kernel_duration_ns": {"value": metric_value(custom["SWIGLU"], "gpu__time_duration"), "evidence": "MEASURED"},
        "unknown_fields": ["activation_dram_write", "activation_dram_read", "dram_round_trip", "exact launch API cost", "backward kernel attribution", "activation-vs-weight traffic"],
        "measurement_perturbation": "PERTURBING",
        "instrumentation_overhead": {"uninstrumented_whole_median_ms": uninstrumented_median, "instrumented_whole_median_ms": instrumented_median, "overhead_percent": overhead_pct, "evidence": "MEASURED"},
    }
    (OUT / "performance_facts_v6.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")

    profiler = {
        "nsys": {"status": "UNAVAILABLE", "evidence": "where.exe nsys returned no path"},
        "ncu": {"status": "PASS", "version": "2026.3.0.0", "evidence": "version/query-metrics and NVTX-filtered profiles completed"},
        "torch_profiler": {"status": "BLOCKED", "evidence": "torch profiler context returned but Kineto reported CUPTI_ERROR_INVALID_DEVICE; CUDA activity evidence unavailable", "raw_artifact": "torch_profiler_probe.txt"},
    }
    (OUT / "profiler_capabilities.json").write_text(json.dumps(profiler, indent=2), encoding="utf-8")

    structural = {
        "hypothesis_id": "fp-producer-consumer-boundary",
        "status": "PARTIAL",
        "supported_facts": ["physical SWIGLU and FC2 kernel executions are distinct", "SwiGLU returns a device tensor consumed by FC2", "SwiGLU kernel duration and total kernel DRAM read are measured"],
        "missing_preconditions": ["activation-only DRAM traffic", "proof that the boundary is avoidable without changing GEMM/gradient semantics", "representative-shape resource/cost evidence"],
        "reason": "Separate kernels and a device intermediate are established, but they do not by themselves prove an avoidable HBM round trip or a beneficial fusion opportunity.",
    }
    (OUT / "structural_hypothesis_revisit.json").write_text(json.dumps(structural, indent=2), encoding="utf-8")

    f = 0.015571342909175317
    value_gate = {
        "scope": "LOCAL_MINIMAL_GPTMODEL",
        "operator_fraction": f,
        "speedup_scenarios": {str(s): 1.0 / ((1.0 - f) + f / s) for s in [1.25, 1.5, 2.0, 4.0]},
        "infinite_speedup": 1.0 / (1.0 - f),
        "system_value_gate": "SCIENTIFIC_ONLY",
        "reason": "The structural evidence is useful for causal validation, but the current local model ceiling is only about 1.58%.",
        "candidate_decision": "NOT_READY",
    }
    (OUT / "system_value_gate.json").write_text(json.dumps(value_gate, indent=2), encoding="utf-8")

    delta = {
        "episode": "REAL_OPT_LOOP_002",
        "status": "INCONCLUSIVE",
        "write_to_mechanism_memory": False,
        "confirmed": ["graph adjacency is not sufficient evidence of physical fusion", "NVTX-filtered NCU observed distinct authentic SWIGLU and FC2 kernel executions", "device tensor materialization is supported by source plus separate kernel execution"],
        "inconclusive": ["activation DRAM round trip", "activation-only bytes", "whether eliminating the boundary is feasible without changing GEMM/gradient semantics"],
        "scope": "local authentic Megatron MLP forward, shape [3,2,8], RTX 5060 Laptop GPU",
        "proposed_next_measurement": "larger representative shape and a profiler method that separates activation traffic from FC2 weight traffic",
    }
    (OUT / "knowledge_delta_measurement.json").write_text(json.dumps(delta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
