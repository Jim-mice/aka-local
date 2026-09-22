from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "integration" / "swiglu" / "real_loop_003"


def read_rows(name: str) -> list[dict[str, str]]:
    lines = (OUT / name).read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    return list(csv.DictReader(lines[start:]))


def value(name: str, metric: str, suffix: str = ".sum") -> float | None:
    rows = read_rows(name)
    for row in rows:
        if row.get("Metric Name") == metric + suffix:
            try:
                return float(row["Metric Value"].replace(",", ""))
            except ValueError:
                return None
    return None


def main() -> int:
    footprint = json.loads((OUT / "activation_footprint.json").read_text(encoding="utf-8"))
    timing = json.loads((OUT / "timing_attribution.json").read_text(encoding="utf-8"))
    activation_bytes = footprint["activation"]["payload_bytes"]
    baseline_read = value("ncu_fc2_baseline_003.csv", "dram__bytes_op_read")
    pressure_read = value("ncu_fc2_pressure_003.csv", "dram__bytes_op_read")
    baseline_l2 = value("ncu_fc2_baseline_003.csv", "lts__t_bytes")
    pressure_l2 = value("ncu_fc2_pressure_003.csv", "lts__t_bytes")
    baseline_us = (value("ncu_fc2_baseline_003.csv", "gpu__time_duration") or 0.0) / 1000.0
    pressure_us = (value("ncu_fc2_pressure_003.csv", "gpu__time_duration") or 0.0) / 1000.0
    cache = {
        "status": "PASS",
        "l2_cache_size_bytes": footprint["environment"]["l2_cache_size_bytes"],
        "pressure_buffer_bytes": footprint["environment"]["l2_cache_size_bytes"] * 2,
        "baseline": {"fc2_duration_us": baseline_us, "fc2_dram_read_bytes_total": baseline_read, "fc2_l2_request_bytes_total": baseline_l2},
        "pressure": {"fc2_duration_us": pressure_us, "fc2_dram_read_bytes_total": pressure_read, "fc2_l2_request_bytes_total": pressure_l2},
        "delta": {"fc2_duration_us": pressure_us - baseline_us, "fc2_dram_read_bytes_total": pressure_read - baseline_read, "fc2_l2_request_bytes_total": pressure_l2 - baseline_l2},
        "interpretation": "No stable increase in FC2 duration, total DRAM reads, or L2 requests under a 2x-L2 pressure buffer; this is indirect evidence only and does not measure activation-only cache hits.",
        "evidence_class": "INFERRED",
    }
    (OUT / "fc2_cache_pressure_comparison.json").write_text(json.dumps(cache, indent=2), encoding="utf-8")

    launch = {
        "swiglu_grid": [1, 1, 1],
        "swiglu_block": [128, 1, 1],
        "threads_per_block": 128,
        "warps_per_block_derived": 4,
        "number_of_blocks": 1,
        "theoretical_occupancy": "100% according to NCU basic warning for this tiny kernel",
        "achieved_occupancy": "7.25%",
        "sm_count": 26,
        "low_occupancy_cause": "GRID_UNDERSUBSCRIBED",
        "basis": "one 128-thread block on a 26-SM GPU; 16 registers/thread and zero dynamic/static shared memory do not support a register/shared-memory resource-limit claim",
    }
    (OUT / "launch_geometry_and_occupancy.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")

    facts = {
        "schema": "MeasurementFacts.v3",
        "episode": "REAL_OPT_LOOP_003",
        "exact_workload": footprint["environment"],
        "activation_footprint": footprint["activation"],
        "timing_attribution": timing,
        "launch_geometry": launch,
        "cache_pressure": cache,
        "hierarchy": {
            "GRAPH_VALUE_EXISTS": {"value": True, "evidence": "SOURCE_DERIVED"},
            "DEVICE_TENSOR_MATERIALIZED": {"value": True, "evidence": "SOURCE_DERIVED + separate kernel execution"},
            "SEPARATE_KERNELS": {"value": True, "evidence": "MEASURED by NVTX-filtered NCU"},
            "CACHE_RESIDENCY": {"value": "INCONCLUSIVE", "evidence": "pressure comparison is indirect; no cache-hit metric available"},
            "DRAM_ROUND_TRIP": {"value": "UNKNOWN", "evidence": "activation-only traffic is not isolated"},
        },
        "local_shape_perf_generalization": "UNSAFE",
        "reason": "one-block tiny workload, 384-byte activation, cache-pressure-insensitive FC2, and low model fraction are not representative evidence for large-model fusion decisions",
    }
    (OUT / "measurement_facts_003.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")

    delta = {
        "episode": "REAL_OPT_LOOP_003",
        "status": "REFUTED",
        "write_to_mechanism_memory": False,
        "confirmed": [
            "activation payload is 384 bytes for the frozen local workload",
            "per-call CUDA Event timing is much larger than active NCU kernel duration",
            "outer batched Event amortizes the single-call boundary but remains above active kernel time",
            "low achieved occupancy is consistent with one-block grid undersubscription, not proven register pressure",
            "separate CUDA kernels do not establish an HBM round trip",
        ],
        "refuted_for_local_workload": ["eliminate-HBM-round-trip is not supported as the local SwiGLU-to-FC2 causal story"],
        "inconclusive": ["activation-only cache hit rate", "activation-only DRAM bytes", "generalization to real nine-grid shapes"],
    }
    (OUT / "knowledge_delta_003.json").write_text(json.dumps(delta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
