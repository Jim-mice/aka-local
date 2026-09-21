"""Phase 12.5: NSYS profiler integration — lightweight, no NCU required.

Extends the existing lab/core/diagnostic.py with NSYS-specific evidence extraction.

Design constraints:
- Only profile candidates that already passed compile + correctness.
- Do NOT claim MEMORY_BOUND / COMPUTE_BOUND without direct profiler support.
- Unknown → UNKNOWN is the safe default.
"""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lab.core.diagnostic import Diagnostic, DiagnosticCategory

ROOT = Path(__file__).resolve().parent.parent.parent.parent


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_nsys_config() -> dict:
    """Load NSYS configuration from environment YAML."""
    cfg_path = ROOT / "config" / "environments" / "v100.yaml"
    if not cfg_path.is_file():
        return {}
    import yaml
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("profiler", {})


def run_nsys_on_v100(remote_work_dir: str, candidate_basename: str = "candidate") -> Optional[dict]:
    """Run NSYS profiling on the V100 for a compiled candidate.

    Returns parsed evidence dict or None on failure.
    """
    from lab.runtime.evaluators.remote_v100 import _ssh_run, _scp_download
    import tempfile

    nsys_cfg = load_nsys_config()
    nsys_bin = nsys_cfg.get("nsys_bin", "<REMOTE_HOME>/tools/nsys-2026.2.1/target-linux-x64/nsys")
    reports = nsys_cfg.get("reports", ["cuda_gpu_kern_sum", "cuda_api_sum"])

    evidence = {
        "source": "nsys",
        "nsys_bin": nsys_bin,
        "timestamp": utcnow(),
        "reports": {},
        "raw_summary": {},
    }

    # Run each NSYS stats report
    nsys_rep = f"{remote_work_dir}/nsys_{candidate_basename}.nsys-rep"
    # First check if NSYS report exists
    code, out, err = _ssh_run(f"test -f {nsys_rep}.gz && echo EXISTS || echo MISSING", timeout=10)
    if "MISSING" in out:
        # Try uncompressed
        code, out, err = _ssh_run(f"test -f {nsys_rep} && echo EXISTS || echo MISSING", timeout=10)
        if "MISSING" in out:
            evidence["error"] = "NSYS report not found"
            return evidence

    for report in reports:
        cmd = f"{nsys_bin} stats --report {report} {nsys_rep} 2>/dev/null"
        code, out, err = _ssh_run(cmd, timeout=30)
        if code == 0 and out.strip():
            evidence["reports"][report] = out.strip()
            evidence["raw_summary"][report] = _parse_nsys_report_text(report, out)
        else:
            evidence["reports"][report] = f"ERROR: {err[:200]}" if err else "no output"

    return evidence


def _parse_nsys_report_text(report_name: str, text: str) -> dict:
    """Parse nsys stats text output into structured metrics.

    Reports we handle:
    - cuda_gpu_kern_sum: kernel durations
    - cuda_api_sum: CUDA API call stats
    """
    result = {}
    if report_name == "cuda_gpu_kern_sum":
        # Extract kernel count, total time, per-kernel times
        lines = text.split("\n")
        kernels = []
        total_time_us = 0.0
        in_table = False
        for line in lines:
            if "Time (%)" in line or "Total Time" in line:
                in_table = True
                continue
            if in_table and line.strip():
                # Try to parse kernel line
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        time_pct = float(parts[0])
                        total_ns = float(parts[1].replace(",", ""))
                        inst = int(parts[2].replace(",", ""))
                        avg_ns = float(parts[3].replace(",", ""))
                        min_ns = float(parts[5].replace(",", ""))
                        max_ns = float(parts[6].replace(",", ""))
                        name = " ".join(parts[8:]) if len(parts) > 8 else (parts[7] if len(parts) > 7 else "unknown")
                        kernels.append({
                            "name": name,
                            "time_pct": time_pct,
                            "total_ns": total_ns,
                            "instances": inst,
                            "avg_ns": avg_ns,
                            "min_ns": min_ns,
                            "max_ns": max_ns,
                        })
                        total_time_us += total_ns / 1000.0
                    except (ValueError, IndexError):
                        pass
        result["kernel_count"] = len(kernels)
        result["total_kernel_time_us"] = round(total_time_us, 2)
        result["kernels"] = kernels[:10]  # top 10

    elif report_name == "cuda_api_sum":
        lines = text.split("\n")
        apis = []
        in_table = False
        for line in lines:
            if "Time (%)" in line or "Total Time" in line:
                in_table = True
                continue
            if in_table and line.strip():
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        time_pct = float(parts[0])
                        total_ns = float(parts[1].replace(",", ""))
                        calls = int(parts[2].replace(",", ""))
                        name = " ".join(parts[7:]) if len(parts) > 7 else (parts[6] if len(parts) > 6 else "unknown")
                        apis.append({
                            "name": name,
                            "time_pct": time_pct,
                            "total_ns": total_ns,
                            "calls": calls,
                        })
                    except (ValueError, IndexError):
                        pass
        result["api_count"] = len(apis)
        result["apis"] = apis[:10]

    return result


def generate_diagnostic_from_nsys(
    nsys_evidence: dict,
    benchmark_result: dict,
    operator: str = "",
    episode: int = 0,
) -> Diagnostic:
    """Generate a structured Diagnostic from NSYS evidence + benchmark.

    Conservative: only claims what NSYS directly supports.
    Falls back to UNKNOWN when evidence is insufficient.

    Phase 12.5: Uses NSYS-derived categories (KERNEL_LATENCY, LAUNCH_OVERHEAD,
    MULTI_KERNEL_OVERHEAD, MEMCPY_OVERHEAD, SYNCHRONIZATION_OVERHEAD, UNKNOWN).
    """
    raw = nsys_evidence.get("raw_summary", {})
    gpu_kern = raw.get("cuda_gpu_kern_sum", {})
    api_sum = raw.get("cuda_api_sum", {})

    kernel_count = gpu_kern.get("kernel_count", 0)
    total_kernel_us = gpu_kern.get("total_kernel_time_us", 0)
    geo_mean = benchmark_result.get("geometric_mean_speedup", 1.0)

    evidence_items = []
    recommendations = []
    category = DiagnosticCategory.UNKNOWN
    confidence = 0.3

    # Collect basic evidence always
    if kernel_count > 0:
        evidence_items.append(f"kernel_count={kernel_count}")
        evidence_items.append(f"total_kernel_time_us={total_kernel_us}")
    evidence_items.append(f"geo_mean_speedup={geo_mean}")

    # --- Rule 1: Memcpy detected → MEMCPY_OVERHEAD (check before kernel rules) ---
    memcpy_time_us = 0.0
    for api in api_sum.get("apis", []):
        name = api.get("name", "").lower()
        if "memcpy" in name or "cudamemcpy" in name:
            memcpy_time_us += api.get("total_ns", 0) / 1000.0
    if memcpy_time_us > 0:
        evidence_items.append(f"memcpy_time_us={round(memcpy_time_us, 2)}")
        if memcpy_time_us > total_kernel_us * 0.3:
            category = DiagnosticCategory.MEMCPY_OVERHEAD
            confidence = 0.5
            recommendations.append("Significant memcpy time relative to kernel time")
            recommendations.append("Consider reducing host-device transfers or using page-locked memory")

    # --- Rule 2: Sync calls → SYNCHRONIZATION_OVERHEAD (check before kernel rules) ---
    sync_count = 0
    for api in api_sum.get("apis", []):
        name = api.get("name", "").lower()
        if "sync" in name or "cudadevicesynchronize" in name:
            sync_count += api.get("calls", 0)
    if sync_count > 0:
        evidence_items.append(f"sync_calls={sync_count}")
        if sync_count > 2 and category == DiagnosticCategory.UNKNOWN:
            category = DiagnosticCategory.SYNCHRONIZATION_OVERHEAD
            confidence = 0.5
            recommendations.append(f"Detected {sync_count} synchronization calls — consider reducing sync points")

        # --- Compute dominant kernel percentage ---
    dominant_kernel_pct = 0.0
    dominant_kernel_name = ""
    for k in gpu_kern.get("kernels", []):
        pct = k.get("time_pct", 0)
        if pct > dominant_kernel_pct:
            dominant_kernel_pct = pct
            dominant_kernel_name = k.get("name", "")
    if dominant_kernel_pct > 0:
        evidence_items.append(f"dominant_kernel_pct={dominant_kernel_pct}")

    # --- Rule 3: Multiple kernel launches (no dominant kernel)  → MULTI_KERNEL_OVERHEAD ---
    if kernel_count > 1 and geo_mean < 1.0 and category == DiagnosticCategory.UNKNOWN and dominant_kernel_pct < 95.0:
        category = DiagnosticCategory.MULTI_KERNEL_OVERHEAD
        confidence = 0.55
        evidence_items.append("multi_kernel_detected=true")
        recommendations.append("Multiple kernel launches detected — consider kernel fusion")
        recommendations.append("Each launch adds API overhead on V100")
        # Check for CUDA launch API
        cuda_launch_us = 0.0
        for api in api_sum.get("apis", []):
            if "launch" in api.get("name", "").lower():
                cuda_launch_us += api.get("total_ns", 0) / 1000.0
        if cuda_launch_us > 0:
            evidence_items.append(f"cuda_launch_api_time_us={round(cuda_launch_us, 2)}")
            recommendations.append("CUDA launch API overhead confirmed in nsys output")

    # --- Rule 4: Single kernel, below baseline → KERNEL_LATENCY ---
    if (kernel_count == 1 or dominant_kernel_pct >= 95.0) and geo_mean < 1.0 and category == DiagnosticCategory.UNKNOWN:
        category = DiagnosticCategory.KERNEL_LATENCY
        confidence = 0.6
        evidence_items.append("single_kernel=true")
        recommendations.append("Kernel latency is the bottleneck — single kernel is slower than reference")
        recommendations.append("Focus on instruction-level optimization within the kernel")
        recommendations.append("Investigate reduction efficiency and memory access patterns")

    # --- Rule 6: Single kernel (or dominant), at or above baseline → insufficient for diagnosis ---
    if (kernel_count == 1 or dominant_kernel_pct >= 95.0) and geo_mean >= 1.0 and category == DiagnosticCategory.UNKNOWN:
        evidence_items.append("single_kernel_at_or_above_baseline=true")
        recommendations.append("Kernel performs at or above baseline — no performance bottleneck identified from NSYS")

    # --- Fallback ---
    if not evidence_items:
        evidence_items.append("insufficient_nsys_data")
        recommendations.append("Collect more detailed profiling data")

    experiment_id = f"{operator}_episode_{episode}" if operator else ""

    return Diagnostic(
        category=category,
        confidence=min(confidence, 0.9),
        message=f"NSYS-derived diagnosis for {experiment_id}",
        evidence=evidence_items,
        possible_causes=[
            "Launch overhead from multiple kernel invocations",
            "Suboptimal kernel implementation",
            "Synchronization overhead",
            "Memory transfer overhead",
        ],
        suggested_actions=recommendations,
        source="nsys_profile",
        experiment_id=experiment_id,
    )


def save_diagnostic(diagnostic: Diagnostic, ep_dir: Path) -> Path:
    """Save diagnostic.json to episode directory."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    path = ep_dir / "diagnostic.json"
    data = diagnostic.to_dict()
    data["generated_at"] = utcnow()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_profile_context(operator: str, environment: str = "v100_sm70") -> dict:
    """Build profile context for agent prompt from most recent valid episode."""
    camp_dir = ROOT / "campaigns" / operator
    if not camp_dir.is_dir():
        return {}

    # Find most recent episode with a valid result
    best_episode = None
    best_score = float("-inf")
    best_diag = None

    for ep_dir in sorted(camp_dir.iterdir()):
        if not ep_dir.name.startswith("episode_"):
            continue
        result_path = ep_dir / "result.json"
        diag_path = ep_dir / "diagnostic.json"
        dec_path = ep_dir / "decision.json"

        if not result_path.is_file():
            continue

        result = _read_json(result_path)
        if not result:
            continue
        if not result.get("compile_pass") or not result.get("correctness_pass"):
            continue

        score = result.get("geometric_mean_speedup", 0)
        if score > best_score:
            best_score = score
            best_episode = ep_dir
            if diag_path.is_file():
                best_diag = _read_json(diag_path)

    if best_episode is None:
        return {}

    ep_num = best_episode.name
    result = _read_json(best_episode / "result.json") or {}
    decision = _read_json(best_episode / "decision.json") or {}

    context = {
        "episode": ep_num,
        "geo_mean_speedup": best_score,
        "compile_pass": result.get("compile_pass"),
        "correctness_pass": result.get("correctness_pass"),
        "shapes": [
            {"shape": s.get("shape"), "latency_us": s.get("latency_us")}
            for s in result.get("shapes", [])
        ],
        "profile_available": result.get("profile_available", False),
        "diagnostic": best_diag,
    }

    return context


def _read_json(path):
    if not path.is_file():
        return None
    for enc in ["utf-8", "utf-8-sig"]:
        try:
            return json.loads(path.read_text(encoding=enc))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return None


def validate_diagnostic(diagnostic: Diagnostic) -> tuple[bool, list[str]]:
    """Validate that a diagnostic's category is supported by its evidence.

    Phase 12.5 (Task 6): Each category requires specific evidence.
    Unsupported diagnoses automatically degrade to UNKNOWN.

    Returns (is_valid, issues).
    """
    cat = diagnostic.category
    evidence = diagnostic.evidence
    if isinstance(evidence, dict):
        ev_items = evidence.get("items", [])
    elif isinstance(evidence, list):
        ev_items = evidence
    else:
        ev_items = []

    ev_text = " ".join(str(e) for e in ev_items)
    issues = []

    if cat == DiagnosticCategory.KERNEL_LATENCY:
        has_single = "kernel_count=1" in ev_text
        has_dominant = "dominant_kernel_pct=" in ev_text
        if not has_single and not has_dominant:
            issues.append("KERNEL_LATENCY requires kernel_count=1 or dominant_kernel_pct evidence")
        if "geo_mean_speedup" not in ev_text:
            issues.append("KERNEL_LATENCY requires geo_mean_speedup evidence")

    elif cat == DiagnosticCategory.MULTI_KERNEL_OVERHEAD:
        has_multi = any("kernel_count=" in e and not e.endswith("=1") and not e.endswith("=0") for e in ev_items)
        if not has_multi and "multi_kernel" not in ev_text:
            issues.append("MULTI_KERNEL_OVERHEAD requires kernel_count > 1 evidence")

    elif cat == DiagnosticCategory.LAUNCH_OVERHEAD:
        if "launch" not in ev_text.lower():
            issues.append("LAUNCH_OVERHEAD requires launch/API overhead evidence")

    elif cat == DiagnosticCategory.MEMCPY_OVERHEAD:
        if "memcpy" not in ev_text.lower():
            issues.append("MEMCPY_OVERHEAD requires memcpy evidence")

    elif cat == DiagnosticCategory.SYNCHRONIZATION_OVERHEAD:
        if "sync" not in ev_text.lower():
            issues.append("SYNCHRONIZATION_OVERHEAD requires synchronization evidence")

    elif cat in (DiagnosticCategory.MEMORY_BOUND, DiagnosticCategory.COMPUTE_BOUND,
                  DiagnosticCategory.REGISTER_PRESSURE, DiagnosticCategory.OCCUPANCY_LIMITED):
        issues.append(f"{cat.value} requires NCU hardware profile evidence, not NSYS")

    return len(issues) == 0, issues


def degrade_unsupported_diagnostic(diagnostic: Diagnostic) -> Diagnostic:
    """Degrade a diagnostic to UNKNOWN if evidence is insufficient.

    Returns a new Diagnostic (or the same one if valid).
    """
    valid, issues = validate_diagnostic(diagnostic)
    if valid:
        return diagnostic

    # Degrade to UNKNOWN
    return Diagnostic(
        category=DiagnosticCategory.UNKNOWN,
        confidence=0.2,
        severity=diagnostic.severity,
        message=f"Degraded from {diagnostic.category.value}: {'; '.join(issues)}",
        evidence=diagnostic.evidence,
        possible_causes=diagnostic.possible_causes,
        suggested_actions=diagnostic.suggested_actions,
        source=diagnostic.source,
        experiment_id=diagnostic.experiment_id,
    )
