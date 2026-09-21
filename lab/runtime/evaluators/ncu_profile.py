"""Small, evidence-only parser for NCU CSV output.

It intentionally reports measured fields and missing capabilities; it never
labels a kernel memory-bound, launch-bound, or otherwise bottlenecked.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path


BASIC_DIAGNOSTIC_METRICS = (
    "gpu__time_duration.avg",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)


def parse_ncu_csv(text: str) -> dict:
    """Parse available metric rows across NCU CSV layouts without guessing units."""
    rows = [[item.strip() for item in row] for row in csv.reader(io.StringIO(text)) if len(row) >= 2]
    metrics, kernel_name = {}, None
    # NCU 2026 raw CSV has a header row, a unit row, then one or more data
    # rows.  Decode that table first; fall back to simple metric/value rows for
    # fixtures and older NCU output layouts.
    for index, header in enumerate(rows):
        if "Kernel Name" not in header or not any("__" in item for item in header):
            continue
        for data in rows[index + 1:]:
            if len(data) != len(header) or "rmsnorm_row_kernel" not in " ".join(data):
                continue
            table = dict(zip(header, data))
            kernel_name = table.get("Kernel Name") or kernel_name
            for name, value in table.items():
                if ("__" in name or name in {"Block Size", "Grid Size"}) and value not in {"", "N/A", "n/a"}:
                    metrics[name] = value
            return {"kernel_name": kernel_name, "metrics": metrics}
    for row in rows:
        joined = " ".join(row).lower()
        if "rmsnorm_row_kernel" in joined:
            kernel_name = next((item for item in row if "rmsnorm_row_kernel" in item), kernel_name)
        name = next((item for item in row if "__" in item), None)
        if name:
            value = next((item for item in reversed(row) if item not in {"", "N/A", "n/a"}), None)
            if value and value != name:
                metrics[name] = value
    return {"kernel_name": kernel_name, "metrics": metrics}


def parsed_profile(worker_metadata: dict, ncu_csv: str, *, static_evidence: dict | None = None) -> dict:
    parsed = parse_ncu_csv(ncu_csv)
    metrics = parsed["metrics"]
    missing = [name for name in BASIC_DIAGNOSTIC_METRICS if name not in metrics]
    static = static_evidence or {}
    return {
        "kind": "PROFILE_MEASUREMENT",
        "not_latency_benchmark": True,
        "shape_id": worker_metadata.get("shape_id"),
        "token_count": worker_metadata.get("token_count"),
        "hidden_size": worker_metadata.get("hidden_size"),
        "kernel": {"name": parsed.get("kernel_name") or worker_metadata.get("expected_kernel_symbol"), "duration": metrics.get("gpu__time_duration.avg")},
        "launch": {"grid": metrics.get("Grid Size") or metrics.get("launch__grid_dim_x"), "block": metrics.get("Block Size") or metrics.get("launch__block_dim_x"), "threads_per_block": metrics.get("launch__block_size") or metrics.get("launch__thread_count")},
        "resources": {"registers_per_thread": metrics.get("launch__registers_per_thread") or static.get("registers_per_thread"), "shared_memory_bytes": metrics.get("launch__shared_mem_per_block") or static.get("shared_memory_per_block")},
        "occupancy": {"achieved": metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active"), "theoretical": None},
        "memory": {"dram_throughput_pct_of_peak_elapsed": metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed")},
        "warp": {"active_warps_pct_of_peak": metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active")},
        "raw_metrics": metrics,
        "capability_missing": missing + ["launch/API overhead requires Nsight Systems; kernel duration is not end-to-end launch overhead"],
    }


def write_profile_summary(path: Path, parsed: dict, *, static_source: str | None = None) -> None:
    content = "\n".join((
        f"# Shape {parsed.get('shape_id')} — RMSNorm NCU profile evidence", "",
        "`PROFILE_MEASUREMENT` · `NOT_LATENCY_BENCHMARK`", "",
        f"- token_count: `{parsed.get('token_count')}`", f"- hidden_size: `{parsed.get('hidden_size')}`", f"- kernel: `{(parsed.get('kernel') or {}).get('name')}`", f"- kernel duration: `{(parsed.get('kernel') or {}).get('duration')}`", f"- registers/thread (static source): `{(parsed.get('resources') or {}).get('registers_per_thread')}`", f"- shared memory/block (static source): `{(parsed.get('resources') or {}).get('shared_memory_bytes')}`", f"- achieved occupancy / active warps: `{(parsed.get('occupancy') or {}).get('achieved')}`", f"- DRAM throughput: `{(parsed.get('memory') or {}).get('dram_throughput_pct_of_peak_elapsed')}`", f"- static source: `{static_source}`", "", "## Missing capability", "", *[f"- {item}" for item in parsed.get("capability_missing", [])], "", "This summary states measured facts only. It does not classify a bottleneck or establish an optimization direction.", "",
    ))
    path.write_text(content, encoding="utf-8")
