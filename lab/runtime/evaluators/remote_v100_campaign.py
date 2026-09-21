"""Phase 8-C: V100 Remote Campaign Runner with Correctness Guarantees.

- Incumbent scoring via incumbent_manifest.json (multi-shape only)
- Frozen evaluation contract from config/
- NSYS graceful hierarchy (nsys stats -> raw report -> unavailable)
- Knowledge summarization for scalable agent context
- lab doctor self-test command
"""

import json
import math
import os
import re
import shutil
import statistics
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from lab.runtime.evaluators.phase8d import (replay_episode, build_episode_manifest, list_operators, load_operator_meta, collect_environment_snapshot, file_sha256, sha256_hex)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# -- Frozen evaluation contract --
EVAL_CONTRACT_PATH = ROOT / "config" / "environments" / "v100_sm70" / "evaluation.json"


def load_evaluation_contract(operator: str = None) -> dict:
    """Load frozen evaluation contract. Falls back to defaults.
    If operator is given, checks for operator-specific config first.
    """
    # Try operator-specific config
    if operator:
        op_path = ROOT / "config" / "environments" / "v100_sm70" / f"evaluation_{operator}.json"
        if op_path.is_file():
            for enc in ["utf-8", "utf-8-sig"]:
                try:
                    return json.loads(op_path.read_text(encoding=enc))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
    # Fall back to default
    if EVAL_CONTRACT_PATH.is_file():
        for enc in ["utf-8", "utf-8-sig"]:
            try:
                return json.loads(EVAL_CONTRACT_PATH.read_text(encoding=enc))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    return {
        "shapes": [[4, 4096], [1, 4096], [8, 4096]],
        "score": "geometric_mean_speedup",
        "correctness_tolerance": 0.001,
    }


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    for enc in ["utf-8", "utf-8-sig"]:
        try:
            return json.loads(path.read_text(encoding=enc))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return default


def debug(msg: str) -> None:
    print(f"[V100] {msg}", flush=True)


# ============================================================
# Phase 8-C: Incumbent Manifest
# ============================================================

def load_incumbent_manifest(operator: str) -> dict:
    """Load the incumbent manifest. Returns defaults if not found."""
    path = ROOT / "campaigns" / operator / "incumbent_manifest.json"
    if path.is_file():
        return read_json(path, {})
    return {
        "operator": operator,
        "incumbent": None,
        "score_type": load_evaluation_contract()["score"],
        "shapes": [",".join(str(x) for x in s) for s in load_evaluation_contract()["shapes"]],
        "score": 1.0,
        "created_from_episode": 0,
    }


def save_incumbent_manifest(operator: str, episode_num: int, score: float, gm: float) -> None:
    """Save incumbent manifest on promotion."""
    contract = load_evaluation_contract()
    manifest = {
        "operator": operator,
        "incumbent": f"v{episode_num}",
        "score_type": contract["score"],
        "shapes": [",".join(str(x) for x in s) for s in contract["shapes"]],
        "score": score,
        "geometric_mean_speedup": gm,
        "created_from_episode": episode_num,
        "updated_at": utcnow(),
    }
    path = ROOT / "campaigns" / operator / "incumbent_manifest.json"
    write_json(path, manifest)
    debug(f"  Incumbent updated: v{episode_num} score={score}")


def get_incumbent_score(operator: str) -> float:
    """Get incumbent score from manifest (multi-shape only)."""
    manifest = load_incumbent_manifest(operator)
    return float(manifest.get("score", 1.0))


# ============================================================
# Phase 8-C: Knowledge Summarization
# ============================================================

def build_knowledge_summary(operator: str, environment: str = "v100_sm70") -> dict:
    """Build a compact knowledge summary from all accumulated cards."""
    env_dir = ROOT / "knowledge" / "environments" / environment
    summary = {
        "environment": environment,
        "operator": operator,
        "best_patterns": [],
        "avoid_patterns": [],
        "successful_strategies": [],
        "failed_strategies": [],
        "recommendations": [],
        "confidence": "low",
        "apply_when": [],
        "avoid_when": [],
        "best_score": 1.0,
        "total_accepted": 0,
        "total_rejected": 0,
        "last_updated": utcnow(),
    }

    # Collect best patterns from accepted experiences
    exp_dir = env_dir / operator / "experience"
    if exp_dir.is_dir():
        for f in sorted(exp_dir.glob("*.json")):
            d = read_json(f)
            if not d or d.get("operator") != operator:
                continue
            summary["total_accepted"] += 1
            lesson = d.get("lesson", "")
            result = d.get("result", {})
            sp = result.get("speedup") or result.get("aggregate_score") or 1.0
            if float(sp) > summary["best_score"]:
                summary["best_score"] = float(sp)
            if lesson and len(lesson) > 10:
                summary["best_patterns"].append(lesson[:150])
            ep = d.get("episode", 0)
            summary["successful_strategies"].append({
                "name": lesson[:80] if lesson else f"episode_{ep}",
                "episodes": [ep],
                "average_gain": float(sp),
                "confidence": "medium",
            })

    # Collect avoid patterns from rejected lessons
    les_dir = env_dir / operator / "lessons"
    if les_dir.is_dir():
        for f in sorted(les_dir.glob("*.json")):
            d = read_json(f)
            if not d or d.get("operator") != operator:
                continue
            summary["total_rejected"] += 1
            rule = d.get("reusable_rule", "")
            if rule and len(rule) > 5:
                summary["avoid_patterns"].append(rule)
            ep = d.get("episode", 0)
            summary["failed_strategies"].append({
                "name": (d.get("reason", "") or "unknown")[:80],
                "episodes": [ep],
                "reason": d.get("failure_stage", "unknown"),
            })

    # Deduplicate and trim
    summary["best_patterns"] = list(dict.fromkeys(summary["best_patterns"]))[-5:]
    summary["avoid_patterns"] = list(dict.fromkeys(summary["avoid_patterns"]))[-5:]

    # Build confidence
    total = summary["total_accepted"] + summary["total_rejected"]
    if total >= 15:
        summary["confidence"] = "high"
    elif total >= 5:
        summary["confidence"] = "medium"

    # Build recommendations
    summary["recommendations"] = [
        "Use float4 vectorized loads for coalesced memory access on V100",
        "Use warp-shuffle or shared-memory reduction for normalization statistics",
        "Use one 256-thread block per row for row-wise normalization",
        "Use shared-memory reduction for mean/variance or mean-of-squares",
        "Guard all thread accesses with batch/row bounds checking",
        "Test ALL evaluation shapes before claiming correctness",
    ]

    # Build apply_when / avoid_when
    summary["apply_when"] = [
        "Optimizing normalization kernels on V100 (sm_70)",
        "Batch sizes 1-32 with hidden dim 4096",
        "Standalone CUDA kernel with no PyTorch dependency",
    ]
    summary["avoid_when"] = [
        "Single-shape-only testing (must validate all shapes)",
        "Missing batch/row bounds checking in kernel",
        "Compile failures from syntax errors",
        "Regressions vs current incumbent on ANY shape",
    ]

    return summary
def save_knowledge_summary(operator: str, environment: str = "v100_sm70") -> None:
    """Persist knowledge summary to disk."""
    summary = build_knowledge_summary(operator, environment)
    path = ROOT / "knowledge" / "environments" / environment / operator / "knowledge_summary.json"
    write_json(path, summary)


# ============================================================
# Phase 8-C: Knowledge-injected Agent prompt (scalable)
# ============================================================

def build_knowledge_context(operator: str, environment: str = "v100_sm70") -> str:
    """Phase 8-F: Build knowledge injection block with clear sections."""
    parts = []
    
    parts.append("=== CURRENT TARGET ===")
    incumbent = load_incumbent_manifest(operator)
    parts.append("Incumbent: " + str(incumbent.get("incumbent", "none")))
    parts.append("Score: " + str(incumbent.get("score", 1.0)) + " (" + str(incumbent.get("score_type", "geometric_mean_speedup")) + ")")
    parts.append("Shapes: " + str(incumbent.get("shapes", [])))
    parts.append("")

    parts.append("=== KNOWLEDGE SUMMARY ===")
    ks_path = ROOT / "knowledge" / "environments" / environment / operator / "knowledge_summary.json"
    ks = read_json(ks_path, {})

    # KEY RECOMMENDATIONS (first, most important)
    recs = ks.get("recommendations", [])
    if recs:
        parts.append("--- KEY RECOMMENDATIONS ---")
        for r in recs[:8]:
            parts.append("  - " + str(r))
        parts.append("")

    # WHAT WORKED
    parts.append("--- WHAT WORKED ---")
    successful = ks.get("successful_strategies", [])
    if successful:
        for s in successful[:6]:
            ep_list = s.get("episodes", [])
            parts.append("- " + s["name"] + ": avg_gain=" + str(s.get("average_gain", "?")) + ", episodes=" + str(ep_list) + ", confidence=" + str(s.get("confidence", "?")))
    else:
        summary = build_knowledge_summary(operator, environment)
        for p in summary.get("best_patterns", []):
            parts.append("- " + str(p)[:120])
    parts.append("")

    parts.append("=== WHAT FAILED ===")
    failed = ks.get("failed_strategies", [])
    if failed:
        for f in failed[:5]:
            ep_list = f.get("episodes", [])
            parts.append("- " + str(f.get("name", ""))[:100] + ": episodes=" + str(ep_list) + ", reason=" + str(f.get("reason", "?")))
    else:
        for p in ks.get("avoid_patterns", []):
            parts.append("- " + str(p)[:120])
    parts.append("")
    # Always show avoid patterns from knowledge (separate from failed strategies)
    avoid = ks.get("avoid_patterns", [])
    if avoid:
        parts.append("=== AVOID PATTERNS (from knowledge) ===")
        for p in avoid[:10]:
            parts.append("- " + str(p))
        parts.append("")

    parts.append("=== DO NOT REPEAT ===")
    exp_db = ROOT / "campaigns" / operator / "experiments.jsonl"
    # Fallback to lineage.jsonl if experiments.jsonl missing
    if not exp_db.is_file():
        lineage_db = ROOT / "campaigns" / operator / "lineage.jsonl"
        if lineage_db.is_file():
            exp_db = lineage_db
    if exp_db.is_file():
        recent_failures = []
        for line in exp_db.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("decision", "").startswith("REJECT"):
                recent_failures.append(d)
        recent_failures.sort(key=lambda x: x.get("episode", 0), reverse=True)
        for f in recent_failures[:3]:
            parts.append("- Episode " + str(f["episode"]) + ": " + f["decision"] + " score=" + str(f.get("score", "?")))
    parts.append("")

    parts.append("=== RECENT EXPERIMENTS (last 10) ===")
    if exp_db.is_file():
        all_eps = []
        for line in exp_db.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            all_eps.append(d)
        all_eps.sort(key=lambda x: x.get("episode", 0), reverse=True)
        for e in all_eps[:10]:
            strategies = e.get("strategies", [])
            strat_str = ", ".join(strategies) if strategies else "none"
            parts.append("- Ep " + str(e["episode"]) + ": " + e["decision"] + " score=" + str(e["score"]) + " [" + strat_str + "]")
    parts.append("")

    # Phase 12.5: Profile feedback sections
    try:
        from lab.runtime.evaluators.nsys_profiler import build_profile_context
        profile_ctx = build_profile_context(operator, environment)
        if profile_ctx:
            parts.append("=== PROFILE EVIDENCE ===")
            diag = profile_ctx.get("diagnostic", {})
            if diag:
                evidence = diag.get("evidence", {})
                items = evidence.get("items", []) if isinstance(evidence, dict) else []
                for item in items:
                    parts.append(f"  {item}")
                parts.append("")
            
            parts.append("=== CURRENT DIAGNOSIS ===")
            if diag:
                parts.append(f"  category: {diag.get('category', 'UNKNOWN')}")
                parts.append(f"  confidence: {diag.get('confidence', '?')}")
                parts.append(f"  message: {diag.get('message', '')}")
                actions = diag.get('suggested_actions', [])
                if actions:
                    parts.append("  recommendations:")
                    for a in actions:
                        parts.append(f"    - {a}")
                parts.append("")
            
            parts.append("=== CURRENT PERFORMANCE ===")
            parts.append(f"  geo_mean_speedup: {profile_ctx.get('geo_mean_speedup', '?')}")
            parts.append(f"  episode: {profile_ctx.get('episode', '?')}")
            parts.append("")
            
            parts.append("=== PROFILE-GUIDED RECOMMENDATIONS ===")
            actions = diag.get('suggested_actions', []) if diag else []
            for a in actions:
                parts.append(f"  - {a}")
            if not actions:
                parts.append("  - No specific recommendations from profile data")
            parts.append("")
    except Exception as e:
        pass  # Profile feedback is optional

    return "\n".join(parts)

def _build_v100_prompt(operator, contract_obj, op_type="Normalization"):
    """Build operator-specific V100 agent prompt from structured contract."""
    sig = contract_obj.c_signature()
    roles = contract_obj.prompt_roles_section()
    semantics = contract_obj.prompt_semantics_section()
    marker = contract_obj.contract_marker()
    contract_hash = contract_obj.contract_hash
    contract_version = contract_obj.version
    arg_names_list = contract_obj.arg_names()
    arg_names_str = '", "'.join(arg_names_list)

    # Build hypothesis format as JSON then convert to string representation
    hyp_template = {
        "claim": "what you changed",
        "strategy_tags": ["warp_shuffle_reduction", "shared_memory_reduction"],
        "expected_effects": ["reduce synchronization", "reduce shared memory traffic"],
        "risk": ["register pressure"],
        "operator": operator,
        "contract_version": contract_version,
        "contract_hash": contract_hash,
        "interface": {
            "entry": "launch_kernel",
            "arguments": arg_names_list,
        },
    }
    hyp_format = json.dumps(hyp_template, indent=2)

    return f"""You are optimizing a CUDA {op_type} kernel for NVIDIA Tesla V100 (sm_70).

## Operator
- Name: {operator}
- Type: {op_type} (standalone CUDA, no torch)
- Contract Version: {contract_version}
- Contract Hash: {contract_hash}

## Environment
- GPU: Tesla V100-PCIE-16GB (Volta, compute capability 7.0)
- Architecture: sm_70
- CUDA: 11.8 with nvcc
- Remote Linux server (NOT Windows)
- Standalone CUDA kernel only

## FORBIDDEN - DO NOT CREATE OR USE:
- candidate.py (NO Python files of any kind)
- torch, torch.extension, torch.utils.cpp_extension (NO PyTorch)
- sm_120, sm_100, Blackwell, or any architecture newer than sm_70
- Python imports of any kind
- Any file other than candidate.cu, hypothesis.json, AGENT.md
- CUDART_INF_F (does NOT exist in CUDA 11.8; use -INFINITY or -FLT_MAX or -1.0f/0.0f instead)
- Anonymous namespace around extern C kernel (use file-scope functions only)
- Missing closing braces

## CONTRACT

{sig}

## CONTRACT MARKER (MUST be in candidate.cu)

{marker}

## Argument Semantics

{roles}

## Mathematical Semantics

{semantics}

## Compilation

Compiled with: nvcc -O2 -gencode arch=compute_70,code=sm_70 -shared

## Evaluation
Your kernel will be evaluated across MULTIPLE shapes: 1x4096, 4x4096, 8x4096, 32x4096.
The score is the GEOMETRIC MEAN speedup across all shapes.
A kernel that is fast only at 4x4096 but slow at 1x4096 will score poorly.

## Task
Create exactly THREE files:
1. candidate.cu -- standalone CUDA kernel as described above. MUST contain the CONTRACT MARKER line exactly as shown.
2. hypothesis.json -- MUST follow this EXACT format:

{hyp_format}

strategy_tags MUST use these exact names where applicable:
  warp_shuffle_reduction, shared_memory_optimization, coalesced_memory_access,
  parallel_reduction, fused_multiply_add, reciprocal_sqrt, register_optimization,
  vectorized_loads

3. AGENT.md -- summary of what you changed and why

## Volta (sm_70) Tuning Guide
- 256 threads per block is typical; test 128 and 512
- Use __shfl_down_sync for warp-level reductions
- Avoid bank conflicts in shared memory
- Volta has 64 KB L1/shared memory per SM, 80 SMs, 7.0 TFLOPS FP32
- Performance must be ROBUST across all shapes, not just one

## Rules
- Do NOT run benchmarks or compile CUDA code
- Stop after writing the three files
- Use the EXACT extern "C" void launch_kernel signature shown in CONTRACT above
- Do NOT create any Python files
- hypothesis.json MUST include ALL fields shown: claim, strategy_tags, expected_effects, risk, operator, contract_version, contract_hash, interface
- candidate.cu MUST contain the CONTRACT MARKER line exactly as shown above

Reply only when finished.
"""

def build_agent_prompt(operator: str, environment: str = "v100_sm70") -> str:
    """Build operator-specific agent prompt from structured contract."""
    from lab.core.contract import OperatorContract, contract_from_legacy_metadata
    meta_path = ROOT / "operators" / operator / "metadata.json"
    if meta_path.is_file():
        meta = read_json(meta_path, {})
        # Prefer contract_schema, fall back to legacy
        if "contract_schema" in meta:
            contract_obj = OperatorContract.from_dict(meta["contract_schema"])
        else:
            contract_obj = contract_from_legacy_metadata(meta)
            if contract_obj is None:
                contract_obj = OperatorContract(
                    operator=operator,
                    entry="launch_kernel",
                    arguments=[],
                )
    else:
        contract_obj = OperatorContract(operator=operator, entry="launch_kernel", arguments=[])
    
    op_type = operator.replace("_v100_cuda", "").replace("_", " ").title().replace("Rms Norm", "RMSNorm").replace("Layer Norm", "LayerNorm")
    knowledge = build_knowledge_context(operator, environment)
    return _build_v100_prompt(operator, contract_obj, op_type) + "\n" + knowledge


# ============================================================
# Campaign Logic
# ============================================================

def find_next_episode_v100(operator: str) -> int:
    cd = ROOT / "campaigns" / operator
    existing = set()
    if cd.is_dir():
        for d in cd.iterdir():
            m = re.match(r"^episode_(\d+)$", d.name)
            if m and d.is_dir():
                existing.add(int(m.group(1)))
    return max(existing) + 1 if existing else 1


def find_latest_episode_with_candidate(operator: str) -> int | None:
    cd = ROOT / "campaigns" / operator
    if not cd.is_dir():
        return None
    existing = []
    for d in cd.iterdir():
        m = re.match(r"^episode_(\d+)$", d.name)
        if m and d.is_dir():
            if (d / "candidate.cu").is_file():
                existing.append(int(m.group(1)))
    return max(existing) if existing else None


def get_eval_shapes(operator: str = None) -> list:
    """Get shapes from frozen contract, formatted as 'ROWS,COLS' strings."""
    contract = load_evaluation_contract(operator)
    return [",".join(str(d) for d in s) for s in contract["shapes"]]


def evaluate_v100(candidate_path: Path, shapes: list, operator: str = "rms_norm_v100_cuda", with_profile: bool = True) -> dict:
    """Evaluate candidate.cu on remote V100 with multi-shape."""
    from lab.runtime.evaluators.remote_v100 import evaluate_candidate_multi_shape
    return evaluate_candidate_multi_shape(str(candidate_path), shapes, operator=operator, with_profile=with_profile)


def run_evaluation_for_episode(
    ep_dir: Path,
    episode_num: int,
    operator: str,
    shapes: list,
    with_profile: bool = True,
) -> None:
    """Evaluate candidate.cu with multi-shape scoring from frozen contract."""
    candidate_cu = ep_dir / "candidate.cu"
    if not candidate_cu.is_file():
        debug(f"No candidate.cu in {ep_dir}, skipping evaluation")
        return

    # Phase 8-D: Rich reproducibility bundle
    manifest = build_episode_manifest(episode_num, operator, candidate_cu, shapes)
    write_json(ep_dir / "episode_manifest.json", manifest)

    # Phase 2: Multi-shape evaluation
    debug(f"Phase 2: EVALUATION (shapes={shapes})")
    eval_result = evaluate_v100(candidate_cu, shapes, operator=operator, with_profile=with_profile)

    gm = eval_result.get("geometric_mean_speedup", 1.0)
    debug(f"  compile={eval_result['compile_pass']}, correct={eval_result['correctness_pass']}, geo_mean={gm}")

    write_json(ep_dir / "result.json", eval_result)

    # Phase 8-C Decision: multi-shape score vs incumbent manifest
    incumbent_score = get_incumbent_score(operator)
    aggregate_score = eval_result.get("aggregate_score", 1.0) or 1.0

    accepted = (
        eval_result["compile_pass"]
        and eval_result["correctness_pass"]
        and float(aggregate_score) > incumbent_score
    )

    decision = "ACCEPT" if accepted else "REJECT"
    reason = ""
    compiler_error = ""

    if not eval_result["compile_pass"]:
        reason = eval_result.get("error", "compile failed")
        decision = "REJECT_COMPILE"
        compiler_error = eval_result.get("evidence", {}).get("compile_error", "")
    elif not eval_result["correctness_pass"]:
        reason = eval_result.get("error", "correctness failed")
        decision = "REJECT_CORRECTNESS"
    elif float(aggregate_score) <= incumbent_score:
        reason = f"aggregate_score={aggregate_score} <= incumbent={incumbent_score}"
        decision = "REJECT_PERFORMANCE"

    decision_data = {
        "episode": episode_num, "operator": operator,
        "decision": decision,
        "compile_pass": eval_result["compile_pass"],
        "correctness_pass": eval_result["correctness_pass"],
        "speedup": eval_result.get("speedup"),
        "aggregate_score": aggregate_score,
        "geometric_mean_speedup": eval_result.get("geometric_mean_speedup"),
        "incumbent_score": incumbent_score,
        "reason": reason,
        "evaluated_at": utcnow(),
        "score_type": "geometric_mean_speedup",
        "shapes": shapes,
    }
    write_json(ep_dir / "decision.json", decision_data)

    # Phase 8-C: NSYS profile with graceful hierarchy
    profile_saved = False
    if eval_result.get("profile_available") and eval_result.get("profile_summary"):
        ps = eval_result["profile_summary"]
        env_pro = ROOT / "knowledge" / "environments" / "v100_sm70" / operator / "profiles"
        env_pro.mkdir(parents=True, exist_ok=True)
        profile_card = {
            "episode": episode_num,
            "operator": operator,
            "kernel_time_us": ps.get("kernel_time_us"),
            "launch_overhead_us": ps.get("launch_overhead_us"),
            "memory_behavior": ps.get("memory_behavior", "unknown"),
            "source": ps.get("source", "nsys"),
            "timestamp": utcnow(),
        }
        write_json(env_pro / f"episode_{episode_num}.json", profile_card)
        profile_saved = True
        debug(f"  Profile: kernel_time={ps.get('kernel_time_us','?')}us")
    else:
        # Graceful: write "unavailable" profile card
        env_pro = ROOT / "knowledge" / "environments" / "v100_sm70" / operator / "profiles"
        env_pro.mkdir(parents=True, exist_ok=True)
        profile_card = {
            "episode": episode_num,
            "operator": operator,
            "profile_available": False,
            "reason": (eval_result.get("profile_summary") or {}).get("reason", "nsys unavailable"),
            "timestamp": utcnow(),
        }
        write_json(env_pro / f"episode_{episode_num}.json", profile_card)

    # Phase 8-C: Promotion lineage
    lineage_path = ROOT / "campaigns" / operator / "lineage.jsonl"
    prev_best = load_incumbent_manifest(operator).get("incumbent", "baseline")
    lineage_entry = {
        "episode": episode_num,
        "from": prev_best,
        "to": f"v{episode_num}" if accepted else None,
        "decision": decision,
        "score": aggregate_score,
        "geometric_mean_speedup": gm,
        "incumbent_score": incumbent_score,
        "changed_strategy": _extract_strategy(ep_dir),
        "profile_available": eval_result.get("profile_available", False),
        "timestamp": utcnow(),
    }
    append_jsonl(lineage_path, lineage_entry)

    # Phase 8-C: Knowledge update
    env_knowledge = ROOT / "knowledge" / "environments" / "v100_sm70"
    if accepted:
        # Update incumbent manifest
        save_incumbent_manifest(operator, episode_num, aggregate_score, gm)

        card = {
            "operator": operator, "episode": episode_num,
            "result": {"decision": "ACCEPT", "speedup": eval_result.get("speedup"),
                       "aggregate_score": aggregate_score,
                       "geometric_mean_speedup": gm},
            "lesson": f"Episode {episode_num}: geo_mean={gm}x across {len(shapes)} shapes, score={aggregate_score}",
            "timestamp": utcnow(),
        }
        write_json(env_knowledge / operator / "experience" / f"episode_{episode_num}.json", card)
        debug(f"ACCEPTED: score={aggregate_score} (incumbent was {incumbent_score})")
    else:
        reusable_rule = _derive_reusable_rule(decision, reason, compiler_error)
        card = {
            "operator": operator, "episode": episode_num,
            "environment": "v100_sm70",
            "decision": "REJECT",
            "failure_stage": decision.replace("REJECT_", "").lower(),
            "reason": reason,
            "compiler_error": compiler_error,
            "reusable_rule": reusable_rule,
            "timestamp": utcnow(),
        }
        write_json(env_knowledge / operator / "lessons" / f"episode_{episode_num}.json", card)
        debug(f"REJECTED: {decision} | {reusable_rule}")

    # Update knowledge summary after each episode
    save_knowledge_summary(operator)


def _extract_strategy(ep_dir: Path) -> str:
    hyp = read_json(ep_dir / "hypothesis.json")
    if hyp:
        claim = hyp.get("claim", "")
        if claim:
            return claim[:120]
    agent_md = ep_dir / "AGENT.md"
    if agent_md.is_file():
        text = agent_md.read_text(encoding="utf-8")[:200]
        for line in text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and len(line) > 20:
                return line[:120]
    return "unknown"


def _derive_reusable_rule(decision: str, reason: str, compiler_error: str) -> str:
    if "REJECT_COMPILE" in decision:
        if "undefined" in (compiler_error + reason).lower():
            return "Avoid undefined symbols: ensure all device functions are defined in candidate.cu"
        if "expected" in compiler_error.lower():
            return "Fix syntax errors: check CUDA C syntax before submission"
        return "Avoid compilation errors: ensure candidate.cu is valid standalone CUDA C++"
    if "REJECT_CORRECTNESS" in decision:
        if "nan" in reason.lower() or "inf" in reason.lower():
            return "Avoid NaN/Inf: check division-by-zero and sqrt of negative values"
        return "Avoid correctness failures: verify RMSNorm formula y=(x/rms(x))*weight"
    if "REJECT_PERFORMANCE" in decision:
        return "Avoid regressions: ensure kernel is faster than incumbent across ALL shapes"
    return "Avoid this failure pattern"


# ============================================================
# Phase 8-C: lab doctor
# ============================================================

def run_doctor(environment: str = "v100") -> int:
    """Self-test command. Returns exit code."""
    results = {}
    passed = 0
    failed = 0

    print(f"[{environment.upper()} Doctor]")
    print()

    # 1. SSH reachable
    try:
        from lab.runtime.evaluators.remote_v100 import _ssh_run
        code, out, err = _ssh_run("echo ok", timeout=10)
        results["SSH"] = "PASS" if code == 0 and "ok" in out else "FAIL"
    except Exception as e:
        results["SSH"] = f"FAIL ({e})"

    # 2. CUDA available
    try:
        code, out, err = _ssh_run("/usr/local/cuda-11.8/bin/nvcc --version 2>&1", timeout=10)
        results["CUDA"] = "PASS" if "release" in out.lower() else f"FAIL (got: {out.strip()[:60]})"
    except Exception as e:
        results["CUDA"] = f"FAIL ({e})"

    # 3. GPU available
    try:
        code, out, err = _ssh_run("nvidia-smi --query-gpu=name --format=csv,noheader 2>&1", timeout=10)
        results["GPU"] = "PASS" if "V100" in out else f"FAIL (got: {out.strip()[:50]})"
    except Exception as e:
        results["GPU"] = f"FAIL ({e})"

    # 4. Evaluator: compile reference kernel
    try:
        from lab.runtime.evaluators.remote_v100 import evaluate_candidate
        ref = ROOT / "operators" / "rms_norm_v100_cuda" / "reference.cu"  # default operator for smoke test
        res = evaluate_candidate(str(ref))
        results["Evaluator"] = "PASS" if res["compile_pass"] else f"FAIL ({res.get('error','?')[:60]})"
    except Exception as e:
        results["Evaluator"] = f"FAIL ({e})"

    # 5. Reference kernel correctness + benchmark
    try:
        if res["compile_pass"] and res["correctness_pass"]:
            results["Reference"] = f"PASS (speedup={res['speedup']}x)"
        else:
            results["Reference"] = "FAIL"
    except Exception:
        results["Reference"] = "FAIL"

    # 6. Knowledge isolation (per-operator)
    try:
        v100_ops = set()
        v100_dir = ROOT / "knowledge" / "environments" / "v100_sm70"
        for op_dir in v100_dir.iterdir():
            if op_dir.is_dir() and (op_dir / "experience").is_dir():
                for f in (op_dir / "experience").glob("*.json"):
                    d = read_json(f, {})
                    v100_ops.add(d.get("operator", op_dir.name))
        rtx_ops = set()
        rtx_dir = ROOT / "knowledge" / "environments" / "rtx5060_sm120"
        if rtx_dir.is_dir():
            for op_dir in rtx_dir.iterdir():
                if op_dir.is_dir() and (op_dir / "experience").is_dir():
                    for f in (op_dir / "experience").glob("*.json"):
                        d = read_json(f, {})
                        rtx_ops.add(d.get("operator", op_dir.name))
        overlap = v100_ops & rtx_ops
        results["Knowledge"] = "PASS" if not overlap else f"FAIL (cross-contam: {overlap})"
    except Exception as e:
        results["Knowledge"] = f"FAIL ({e})"

    # 7. Campaign state
    try:
        manifest = load_incumbent_manifest("rms_norm_v100_cuda")  # default for doctor
        if manifest.get("incumbent"):
            results["Campaign"] = f"PASS (incumbent={manifest['incumbent']}, score={manifest['score']})"
        else:
            results["Campaign"] = "PASS (no incumbent yet)"
    except Exception as e:
        results["Campaign"] = f"FAIL ({e})"

    # 8. Evaluation contract
    try:
        contract = load_evaluation_contract()
        if contract.get("shapes"):
            results["Contract"] = f"PASS ({len(contract['shapes'])} shapes)"
        else:
            results["Contract"] = "FAIL (no shapes)"
    except Exception as e:
        results["Contract"] = f"FAIL ({e})"

    # Print results
    for name, status in results.items():
        mark = "[OK]" if status.startswith("PASS") else "[FAIL]"
        print(f"  {mark} {name:20s} {status}")

    passed = sum(1 for v in results.values() if v.startswith("PASS"))
    failed = sum(1 for v in results.values() if not v.startswith("PASS"))
    print(f"\n  {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


# ============================================================
# Main entry
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Phase 8-C V100 Campaign Runner")
    ap.add_argument("--operator", default="rms_norm_v100_cuda")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--skip-agent", action="store_true")
    ap.add_argument("--candidate", type=str, default="")
    ap.add_argument("--no-multi-shape", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--doctor", action="store_true", help="Run self-test")
    ap.add_argument("--replay", type=int, default=0, help="Replay episode N")
    ap.add_argument("--list-operators", action="store_true")
    ap.add_argument("--env", default="v100", help="Environment for doctor")
    args = ap.parse_args()

    if args.doctor:
        sys.exit(run_doctor(args.env))
    if args.replay > 0:
        sys.exit(replay_episode(args.operator, args.replay))
    if args.list_operators:
        for op in list_operators():
            print(f"  {op.get('operator','?')}: {op.get('interface','?')} - {op.get('contract','?')[:80]}")
        return

    operator = args.operator
    campaign_dir = ROOT / "campaigns" / operator
    campaign_dir.mkdir(parents=True, exist_ok=True)
    shapes = get_eval_shapes(args.operator) if not args.no_multi_shape else ["4,4096"]
    with_profile = not args.no_profile

    # --skip-agent with --candidate
    if args.skip_agent and args.candidate:
        cand_path = Path(args.candidate)
        if not cand_path.is_file():
            debug(f"ERROR: candidate file not found: {args.candidate}")
            sys.exit(1)
        episode_num = find_next_episode_v100(operator)
        ep_dir = campaign_dir / f"episode_{episode_num}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(cand_path), str(ep_dir / "candidate.cu"))
        debug(f"=== Manual Evaluation (episode {episode_num}) ===")
        run_evaluation_for_episode(ep_dir, episode_num, operator, shapes, with_profile)
        debug("=== Evaluation complete ===")
        return

    # --skip-agent without --candidate
    if args.skip_agent:
        latest = find_latest_episode_with_candidate(operator)
        if latest is None:
            debug(f"ERROR: No existing episode with candidate.cu in {campaign_dir}")
            sys.exit(1)
        ep_dir = campaign_dir / f"episode_{latest}"
        debug(f"=== Re-evaluating Episode {latest} ===")
        run_evaluation_for_episode(ep_dir, latest, operator, shapes, with_profile)
        debug("=== Evaluation complete ===")
        return

    # Agent-driven loop
    for ep_idx in range(args.episodes):
        episode_num = find_next_episode_v100(operator)
        ep_dir = campaign_dir / f"episode_{episode_num}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        debug(f"=== Episode {episode_num}/{args.episodes} ===")
        debug(f"  Incumbent: {load_incumbent_manifest(operator).get('incumbent','none')} score={get_incumbent_score(operator)}")

        candidate_cu = ep_dir / "candidate.cu"

        if not candidate_cu.is_file():
            debug("Phase 1: AGENT (knowledge-injected)")
            import os as _os
            _os.environ.setdefault("CODEX_HOME", r"<LOCAL_USER_HOME>\.codex")

            prompt = build_agent_prompt(operator, "v100_sm70")
            debug(f"  Prompt: {len(prompt)} chars")

            try:
                from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
                BIN = Path(r"<PROJECT_ROOT>\runtimes\codex-0.154.0\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe")
                config = CodexConfig(codex_bin=str(BIN), cwd=str(ep_dir), client_name="aka_v100", client_title="aka-v100", client_version="0.5", config_overrides=("model_provider=openai",))
                cx = Codex(config)
                try:
                    t = cx.thread_start(model="gpt-5.6-luna", model_provider="openai", cwd=str(ep_dir), sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all, ephemeral=True)
                    t.run(prompt, model="gpt-5.6-luna", effort="low", cwd=str(ep_dir), sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all)
                    debug("Agent complete")
                finally:
                    cx.close()
            except Exception as e:
                debug(f"Agent failed: {e}")
                write_json(ep_dir / "agent_error.json", {"timestamp": utcnow(), "error": str(e)})
                if not candidate_cu.is_file():
                    continue

        if not candidate_cu.is_file():
            debug("No candidate.cu, skipping")
            continue

        debug("Phase 2: EVALUATION")
        run_evaluation_for_episode(ep_dir, episode_num, operator, shapes, with_profile)

    debug(f"=== Campaign complete: {args.episodes} episodes ===")


if __name__ == "__main__":
    main()
