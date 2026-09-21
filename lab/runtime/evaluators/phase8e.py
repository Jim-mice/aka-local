"""Phase 8-E: Statistical benchmarking, strategy intelligence, experiment database."""
import json, statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def read_json(path, default=None):
    if not path.is_file():
        return default
    for enc in ["utf-8", "utf-8-sig"]:
        try:
            return json.loads(path.read_text(encoding=enc))
        except:
            pass
    return default

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")

# ============================================================
# Task 1: Statistical benchmarking via multi-run remote eval
# ============================================================

def evaluate_with_stats(candidate_path, shapes, runs_per_shape=3, operator="rms_norm_v100_cuda"):
    """Multi-run evaluation with mean/std/cv per shape."""
    from lab.runtime.evaluators.remote_v100 import RemoteV100Evaluator, _ssh_run, _scp_upload
    import tempfile, uuid

    candidate_path = str(candidate_path)
    result = {
        "compile_pass": False, "correctness_pass": False,
        "shapes": [], "aggregate_score": 1.0,
        "geometric_mean_speedup": 1.0,
        "speedup": None, "latency_us": None,
        "evidence": {}, "error": "",
        "profile_available": False, "profile_summary": None,
        "statistical": True, "runs_per_shape": runs_per_shape,
    }

    # Compile once
    primary_shape = shapes[0]
    ev = RemoteV100Evaluator(candidate_path, primary_shape, operator)
    if not ev.prepare():
        result["error"] = ev.get_error(); return result
    if not ev.compile():
        result["error"] = ev.get_error()
        result["evidence"] = ev.static_evidence(); return result
    result["compile_pass"] = True
    if not ev.check_correctness():
        result["error"] = ev.get_error()
        result["evidence"] = ev.static_evidence(); return result
    result["correctness_pass"] = True
    result["evidence"] = ev.static_evidence()
    bench = ev.benchmark()
    if bench:
        result["evidence"].update(bench)
    ev.cleanup()

    # Benchmark all shapes with multiple runs
    geo_speedups = []
    for shape in shapes:
        shape_latencies = []
        shape_speedups = []
        for run_idx in range(runs_per_shape):
            sev = RemoteV100Evaluator(candidate_path, shape, operator)
            if not sev.prepare():
                continue
            if not sev.compile():
                continue
            if not sev.check_correctness():
                continue
            bench = sev.benchmark()
            if bench and bench.get("latency_us"):
                shape_latencies.append(float(bench["latency_us"]))
            if bench and bench.get("speedup_vs_torch"):
                shape_speedups.append(float(bench["speedup_vs_torch"]))
            sev.cleanup()

        if shape_latencies:
            mn = round(statistics.mean(shape_latencies), 2)
            st = round(statistics.stdev(shape_latencies), 2) if len(shape_latencies) > 1 else 0.0
            cv = round(st / mn * 100, 1) if mn > 0 else 0.0
            mean_sp = round(statistics.mean(shape_speedups), 3) if shape_speedups else 1.0
        else:
            mn, st, cv, mean_sp = 0, 0, 0, 1.0

        result["shapes"].append({
            "shape": shape,
            "runs": [{"latency_us": l} for l in shape_latencies],
            "mean_latency_us": mn,
            "std_latency_us": st,
            "cv_percent": cv,
            "mean_speedup": mean_sp,
        })
        if mean_sp > 0:
            geo_speedups.append(mean_sp)

    if geo_speedups:
        result["geometric_mean_speedup"] = round(statistics.geometric_mean(geo_speedups), 3)
        result["aggregate_score"] = result["geometric_mean_speedup"]

    primary = result["shapes"][0] if result["shapes"] else {}
    result["speedup"] = primary.get("mean_speedup")
    result["latency_us"] = primary.get("mean_latency_us")

    return result


# ============================================================
# Task 2: Statistical acceptance margin
# ============================================================

DEFAULT_MARGIN = 0.02  # 2% minimum improvement

def decide_with_margin(aggregate_score, incumbent_score, margin=DEFAULT_MARGIN):
    """Statistical acceptance: need margin beyond noise."""
    threshold = incumbent_score * (1.0 + margin)
    improvement_pct = ((aggregate_score - incumbent_score) / incumbent_score * 100) if incumbent_score > 0 else 100
    accepted = aggregate_score > threshold
    return {
        "accepted": accepted,
        "threshold": round(threshold, 3),
        "margin": margin,
        "improvement_percent": round(improvement_pct, 2),
        "reason": f"need >{threshold} (incumbent={incumbent_score} + {margin*100}% margin)" if not accepted else "",
    }


# ============================================================
# Task 4: Strategy-level knowledge extraction
# ============================================================

STRATEGY_PATTERNS = {
    "warp_shuffle_reduction": ["warp shuffle", "shfl_down", "__shfl"],
    "shared_memory_optimization": ["shared memory", "__shared__", "shared_mem"],
    "coalesced_memory_access": ["coalesced", "coalesce"],
    "fused_multiply_add": ["fma", "fmaf", "fused multiply"],
    "reciprocal_sqrt": ["rsqrt", "rsqrtf", "reciprocal"],
    "parallel_reduction": ["reduction", "reduce"],
    "register_optimization": ["register", "registers"],
    "vectorized_loads": ["float4", "vectorized", "vector"],
}

def extract_strategies(lesson_text, agent_md_text=""):
    """Extract named strategies from lesson text and agent notes."""
    combined = (lesson_text + " " + agent_md_text).lower()
    found = []
    for strategy_name, keywords in STRATEGY_PATTERNS.items():
        for kw in keywords:
            if kw in combined:
                found.append(strategy_name)
                break
    return list(set(found))


def build_strategy_knowledge(operator, environment="v100_sm70"):
    """Build strategy-level knowledge from experience cards, lineage, and agent notes."""
    env_dir = ROOT / "knowledge" / "environments" / environment
    exp_dir = env_dir / "experience"
    camp_dir = ROOT / "campaigns" / operator

    strategy_data = {}  # name -> {"episodes": [], "gains": [], "count": 0}
    failed_data = {}

    # Source 1: lineage.jsonl (richest strategy descriptions)
    lineage_path = camp_dir / "lineage.jsonl"
    if lineage_path.is_file():
        for line in lineage_path.read_text(encoding="utf-8").strip().split("\n"):
            d = json.loads(line)
            ep = d.get("episode", 0)
            sp = d.get("score") or d.get("geometric_mean_speedup") or 1.0
            decision = d.get("decision", "")
            strategy_text = d.get("changed_strategy", "")
            
            if decision == "ACCEPT":
                strategies = extract_strategies(strategy_text)
                for s in strategies:
                    if s not in strategy_data:
                        strategy_data[s] = {"episodes": [], "gains": [], "count": 0}
                    strategy_data[s]["episodes"].append(ep)
                    strategy_data[s]["gains"].append(float(sp))
                    strategy_data[s]["count"] += 1
            elif decision.startswith("REJECT"):
                key = strategy_text[:80] if strategy_text else f"episode_{ep}_failure"
                if key not in failed_data:
                    reason = "performance" if "PERFORMANCE" in decision else "compile" if "COMPILE" in decision else "correctness"
                    failed_data[key] = {"episodes": [], "reason": reason}
                failed_data[key]["episodes"].append(ep)

    # Source 2: AGENT.md files (hypothesis claims)
    if camp_dir.is_dir():
        for ep_dir in sorted(camp_dir.glob("episode_*")):
            if not ep_dir.is_dir():
                continue
            try:
                ep_num = int(ep_dir.name.split("_")[-1])
            except ValueError:
                continue
            agent_md = ep_dir / "AGENT.md"
            hyp_json = ep_dir / "hypothesis.json"
            text = ""
            if agent_md.is_file():
                text += agent_md.read_text(encoding="utf-8", errors="replace")[:2000]
            if hyp_json.is_file():
                h = read_json(hyp_json, {})
                text += " " + h.get("claim", "")
            result_json = ep_dir / "result.json"
            sp = 1.0
            if result_json.is_file():
                r = read_json(result_json, {})
                sp = r.get("aggregate_score") or r.get("geometric_mean_speedup") or 1.0
            strategies = extract_strategies(text)
            for s in strategies:
                if s not in strategy_data:
                    strategy_data[s] = {"episodes": [], "gains": [], "count": 0}
                if ep_num not in strategy_data[s]["episodes"]:
                    strategy_data[s]["episodes"].append(ep_num)
                    strategy_data[s]["gains"].append(float(sp))
                    strategy_data[s]["count"] += 1

    # Build successful strategies
    successful = []
    for name, data in strategy_data.items():
        if data["gains"]:
            avg_gain = round(sum(data["gains"]) / len(data["gains"]), 3)
            confidence = "high" if data["count"] >= 2 else "medium" if data["count"] >= 1 else "low"
            successful.append({
                "name": name,
                "episodes": sorted(data["episodes"]),
                "average_gain": avg_gain,
                "count": data["count"],
                "confidence": confidence,
            })
    successful.sort(key=lambda x: x["average_gain"], reverse=True)

    # Build failed strategies
    failed = []
    for key, data in failed_data.items():
        failed.append({
            "name": key,
            "episodes": sorted(data["episodes"]),
            "reason": data["reason"],
        })

    return {
        "operator": operator,
        "successful_strategies": successful,
        "failed_strategies": failed,
        "last_updated": utcnow(),
    }


# ============================================================
# Task 5: Experiment database
# ============================================================

def append_experiment_db(operator, episode_num, decision, score, candidate_path):
    """Append to experiments.jsonl for trajectory analysis."""
    db_path = ROOT / "campaigns" / operator / "experiments.jsonl"

    # Extract strategies from candidate
    strategies = []
    ep_dir = ROOT / "campaigns" / operator / f"episode_{episode_num}"
    agent_md = ep_dir / "AGENT.md"
    hyp = ep_dir / "hypothesis.json"
    text = ""
    if agent_md.is_file():
        text += agent_md.read_text(encoding="utf-8", errors="replace")[:500]
    if hyp.is_file():
        h = read_json(hyp, {})
        text += " " + h.get("claim", "")

    strategies = extract_strategies("", text)

    entry = {
        "episode": episode_num,
        "decision": decision,
        "score": score,
        "strategies": strategies,
        "environment": "v100_sm70",
        "timestamp": utcnow(),
    }
    append_jsonl(db_path, entry)
    return entry
