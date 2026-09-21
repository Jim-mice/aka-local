
"""Phase 9: Production Reliability and Autonomous Optimization."""
import json, hashlib, statistics, sys, os
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
# Task 1: Deterministic Replay
# ============================================================

def build_deterministic_manifest(episode_num, operator, candidate_path, shapes, eval_result=None):
    """Phase 9: Build manifest freezing everything needed for replay."""
    from lab.runtime.evaluators.phase8d import file_sha256
    
    evaluator_hash = "unknown"
    try:
        from lab.runtime.evaluators.remote_v100 import _ssh_run
        code, out, err = _ssh_run("md5sum ~/cuda_kernel_experiments/evaluator/evaluate.py 2>/dev/null | awk '{print $1}'", timeout=10)
        if code == 0 and out.strip():
            evaluator_hash = out.strip()
    except:
        pass
    
    contract_path = ROOT / "config" / "environments" / "v100_sm70" / "evaluation.json"
    contract = read_json(contract_path, {})
    
    manifest = {
        "episode": episode_num,
        "operator": operator,
        "evaluator_hash": evaluator_hash,
        "contract_hash": hashlib.sha256(
            json.dumps({"shapes": shapes}, sort_keys=True).encode()
        ).hexdigest()[:16],
        "candidate_hash": file_sha256(Path(candidate_path)),
        "benchmark_config": {
            "warmup": contract.get("warmup", 50),
            "iterations": contract.get("iterations", 200),
            "compiler": contract.get("compiler", "nvcc"),
            "compiler_flags": contract.get("compiler_flags", "-gencode arch=compute_70,code=sm_70 -O2"),
            "repeat": 3,
        },
        "nvcc_command": "",
        "shapes": shapes,
        "score_type": contract.get("score", "geometric_mean_speedup"),
        "created_time": utcnow(),
    }
    
    if eval_result:
        ev = eval_result.get("evidence", {})
        manifest["nvcc_command"] = ev.get("nvcc_cmd", "")
        manifest["frozen_baselines"] = {
            "torch_naive_latency_us": ev.get("torch_naive_latency_us"),
            "torch_optimized_latency_us": ev.get("torch_optimized_latency_us"),
        }
        manifest["frozen_kernel_latency_us"] = ev.get("latency_us")
    
    return manifest


def deterministic_replay(operator, episode_num):
    """Phase 9: Replay with frozen manifest for determinism."""
    ep_dir = ROOT / "campaigns" / operator / f"episode_{episode_num}"
    candidate = ep_dir / "candidate.cu"
    manifest_path = ep_dir / "episode_manifest.json"
    
    if not candidate.is_file():
        print(f"[FAIL] Candidate not found: {candidate}")
        return 1
    
    manifest = read_json(manifest_path, {})
    shapes = manifest.get("shapes", ["4,4096"])
    frozen_baselines = manifest  # pass full manifest for per-shape support
    candidate_hash = manifest.get("candidate_hash", "")
    
    from lab.runtime.evaluators.phase8d import file_sha256
    current_hash = file_sha256(candidate)
    
    print("[Replay-Deterministic]")
    print(f"  Episode: {episode_num}")
    print(f"  Candidate hash: {current_hash}")
    print(f"  Manifest hash:  {candidate_hash}")
    print(f"  Hash match:     {'PASS' if current_hash == candidate_hash else 'FAIL'}")
    print(f"  Shapes:         {shapes}")
    print(f"  Frozen baselines: naive={frozen_baselines.get('torch_naive_latency_us')}us, opt={frozen_baselines.get('torch_optimized_latency_us')}us")
    print()
    
    orig_result = read_json(ep_dir / "result.json", {})
    orig_score = orig_result.get("geometric_mean_speedup")
    print(f"  Original score:  {orig_score}")
    
    from lab.runtime.evaluators.phase8e import evaluate_with_stats
    result = evaluate_with_stats(str(candidate), shapes, runs_per_shape=3)
    
    if not result.get("compile_pass"):
        print(f"[FAIL] Compile failed: {result.get('error')}")
        return 1
    if not result.get("correctness_pass"):
        print(f"[FAIL] Correctness failed: {result.get('error')}")
        return 1
    
    replay_score = _compute_score_with_frozen_baselines(result, manifest, shapes)
    
    print(f"  Replay score (raw):    {result.get('geometric_mean_speedup')}")
    print(f"  Replay score (frozen): {replay_score}")
    
    if orig_score and replay_score:
        diff_pct = abs(replay_score - orig_score) / orig_score * 100
        issues = []
        if current_hash != candidate_hash:
            issues.append("candidate hash mismatch")
        if diff_pct > 2.0:
            issues.append(f"score deviation {diff_pct:.1f}%")
        
        if not issues:
            print(f"  Difference: {diff_pct:.1f}%")
            print(f"\n  Verdict: PASS (deterministic)")
        elif diff_pct < 5.0:
            print(f"  Difference: {diff_pct:.1f}%")
            print(f"\n  Verdict: WARNING ({'; '.join(issues)})")
        else:
            print(f"  Difference: {diff_pct:.1f}%")
            print(f"\n  Verdict: FAIL ({'; '.join(issues)})")
    else:
        print(f"\n  Verdict: PASS (no original)")
    
    result["frozen_baselines_used"] = bool(frozen_baselines)
    result["frozen_score"] = replay_score
    write_json(ep_dir / "replay_result.json", result)
    
    print(f"\n  Per-shape latencies:")
    for s in result.get("shapes", []):
        print(f"    {s['shape']}: mean={s['mean_latency_us']}us std={s['std_latency_us']}us cv={s['cv_percent']}%")
    
    return 0


def _compute_score_with_frozen_baselines(result, frozen_baselines, shapes):
    """Compute speedup using per-shape frozen baselines."""
    # Try per_shape baselines first (Phase 9+)
    per_shape = frozen_baselines.get("frozen_per_shape_baselines", {}) if frozen_baselines else {}
    
    if per_shape:
        speedups = []
        for s in result.get("shapes", []):
            shape_key = s.get("shape", "")
            frozen = per_shape.get(shape_key, {})
            frozen_torch = frozen.get("torch_optimized_latency_us")
            latency = s.get("mean_latency_us", 0)
            if latency and latency > 0 and frozen_torch and frozen_torch > 0:
                speedups.append(frozen_torch / latency)
        if speedups:
            return round(statistics.geometric_mean(speedups), 3)
    
    # Fallback: single frozen baseline
    if frozen_baselines and frozen_baselines.get("torch_optimized_latency_us"):
        frozen_torch = float(frozen_baselines["torch_optimized_latency_us"])
        speedups = []
        for s in result.get("shapes", []):
            latency = s.get("mean_latency_us", 0)
            if latency and latency > 0 and frozen_torch > 0:
                speedups.append(frozen_torch / latency)
        if speedups:
            return round(statistics.geometric_mean(speedups), 3)
    
    return result.get("geometric_mean_speedup", 1.0)
def build_recommendations(operator, environment="v100_sm70"):
    """Phase 9: Generate actionable recommendations from knowledge."""
    ks_path = ROOT / "knowledge" / "environments" / environment / "knowledge_summary.json"
    ks = read_json(ks_path, {})
    successful = ks.get("successful_strategies", [])
    failed = ks.get("failed_strategies", [])
    
    total = max(len(successful) + len(failed), 1)
    
    recommendations = []
    for s in successful:
        count = s.get("count", 1)
        recommendations.append({
            "name": s["name"],
            "confidence": min(0.95, 0.5 + count / total),
            "success_rate": round(count / total, 2),
            "episodes": s.get("episodes", []),
            "average_gain": s.get("average_gain", 0),
            "apply_when": _derive_apply_when(s["name"]),
            "avoid_when": _derive_avoid_when(s["name"]),
        })
    
    recommendations.sort(key=lambda x: x["confidence"] * x["average_gain"], reverse=True)
    
    avoided = []
    for f in failed:
        avoided.append({
            "name": (f.get("name", "") or "unknown")[:80],
            "reason": f.get("reason", "performance regression"),
            "episodes": f.get("episodes", []),
        })
    
    return {
        "recommendations": recommendations,
        "avoided_searches": avoided,
        "last_updated": utcnow(),
    }


def _derive_apply_when(strategy_name):
    mapping = {
        "warp_shuffle_reduction": ["reduction-dominated kernels", "large thread counts"],
        "shared_memory_optimization": ["reduction operations", "data reuse within block"],
        "coalesced_memory_access": ["memory-bound kernels", "large hidden dimensions"],
        "parallel_reduction": ["element-wise reduction", "multi-stage operations"],
        "fused_multiply_add": ["arithmetic-heavy kernels", "multiply-accumulate patterns"],
        "reciprocal_sqrt": ["normalization layers", "RMS/LayerNorm variants"],
        "register_optimization": ["register-pressure scenarios", "small block sizes"],
        "vectorized_loads": ["wide memory access", "float4-compatible layouts"],
    }
    return mapping.get(strategy_name, ["evaluate on target hardware"])


def _derive_avoid_when(strategy_name):
    mapping = {
        "warp_shuffle_reduction": ["high register pressure", "small warp sizes (< 32)"],
        "shared_memory_optimization": ["no data reuse", "exceeding SM shared memory"],
        "coalesced_memory_access": ["irregular access patterns"],
        "parallel_reduction": ["trivial operations (overhead > benefit)"],
        "fused_multiply_add": ["non-arithmetic kernels"],
        "reciprocal_sqrt": ["high-precision sqrt required"],
        "register_optimization": ["occupancy already saturated"],
        "vectorized_loads": ["non-aligned memory", "odd dimensions"],
    }
    return mapping.get(strategy_name, ["test before adopting"])


# ============================================================
# Task 3: Hypothesis Validation
# ============================================================

REQUIRED_HYPOTHESIS_FIELDS = {
    "strategy_tags": list,
    "expected_effects": list,
    "risk": list,
}

ALLOWED_STRATEGY_TAGS = {
    "warp_shuffle_reduction", "shared_memory_optimization", "coalesced_memory_access",
    "parallel_reduction", "fused_multiply_add", "reciprocal_sqrt",
    "register_optimization", "vectorized_loads",
}


def validate_hypothesis(ep_dir):
    """Phase 9: Validate hypothesis.json schema."""
    if isinstance(ep_dir, str):
        ep_dir = Path(ep_dir)
    hyp_path = ep_dir / "hypothesis.json"
    
    if not hyp_path.is_file():
        return False, ["hypothesis.json not found"]
    
    hyp = read_json(hyp_path)
    if not hyp:
        return False, ["hypothesis.json is empty or invalid JSON"]
    
    errors = []
    for field, expected_type in REQUIRED_HYPOTHESIS_FIELDS.items():
        if field not in hyp:
            errors.append(f"missing required field: {field}")
        elif not isinstance(hyp[field], expected_type):
            errors.append(f"field {field} must be a list, got {type(hyp[field]).__name__}")
    
    tags = hyp.get("strategy_tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if tag not in ALLOWED_STRATEGY_TAGS:
                errors.append(f"unknown strategy_tag: '{tag}'")
    
    effects = hyp.get("expected_effects", [])
    if isinstance(effects, list) and len(effects) < 1:
        errors.append("expected_effects must have at least one entry")
    
    risks = hyp.get("risk", [])
    if isinstance(risks, list) and len(risks) < 1:
        errors.append("risk must have at least one entry")
    
    return (len(errors) == 0), errors


def write_hypothesis_validation_lesson(operator, episode_num, errors):
    """Write lesson when hypothesis validation fails."""
    env_knowledge = ROOT / "knowledge" / "environments" / "v100_sm70"
    card = {
        "operator": operator,
        "episode": episode_num,
        "environment": "v100_sm70",
        "decision": "REJECT",
        "failure_stage": "hypothesis_invalid",
        "reason": "hypothesis.json schema validation failed",
        "validation_errors": errors,
        "reusable_rule": "Ensure hypothesis.json has valid strategy_tags[], expected_effects[], and risk[]",
        "timestamp": utcnow(),
    }
    write_json(env_knowledge / "lessons" / f"{operator}_rejected_episode_{episode_num}.json", card)
    return card


# ============================================================
# Task 4: Unattended Campaign Mode
# ============================================================


# ============================================================
# Log Rotation (Phase 9.5)
# ============================================================

MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB

def rotate_log_if_needed(log_path_str):
    """Rotate continuous_run.log if it exceeds MAX_LOG_SIZE."""
    log_path = Path(log_path_str) if isinstance(log_path_str, str) else log_path_str
    if not log_path.is_file():
        return
    size = log_path.stat().st_size
    if size > MAX_LOG_SIZE:
        backup = log_path.with_suffix('.log.1')
        if backup.is_file():
            backup.unlink()
        log_path.rename(backup)
        log_path.write_text(f"[{utcnow()}] Log rotated (was {size} bytes)\n", encoding="utf-8")

def run_unattended_campaign(operator, env_name, total_episodes, resume=False, candidate_path=None):
    """Phase 9: Run unattended campaign with resume support."""
    import shutil
    from lab.runtime.evaluators.remote_v100_campaign import (
        get_eval_shapes, find_next_episode_v100,
        run_evaluation_for_episode,
    )
    
    log_path = ROOT / "continuous_run.log"
    rotate_log_if_needed(log_path)
    
    def log(msg):
        line = f"[{utcnow()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    
    log(f"=== Campaign: {operator} x{total_episodes}, resume={resume} ===")
    
    shapes = get_eval_shapes(operator)
    camp_dir = ROOT / "campaigns" / operator
    camp_dir.mkdir(parents=True, exist_ok=True)
    
    completed = 0
    failed = 0
    
    for ep_idx in range(total_episodes):
        episode_num = find_next_episode_v100(operator)
        ep_dir = camp_dir / f"episode_{episode_num}"
        
        # Resume: skip if already completed
        decision_path = ep_dir / "decision.json"
        if resume and decision_path.is_file():
            decision = read_json(decision_path, {})
            if decision.get("decision"):
                log(f"Ep {episode_num}: SKIP (completed: {decision['decision']})")
                completed += 1
                continue
        
        ep_dir.mkdir(parents=True, exist_ok=True)
        log(f"--- Ep {episode_num}/{total_episodes} ---")
        
        # Copy candidate if provided
        if candidate_path and Path(candidate_path).is_file():
            shutil.copy2(candidate_path, str(ep_dir / "candidate.cu"))
            log(f"  Using candidate: {candidate_path}")
        
        candidate_cu = ep_dir / "candidate.cu"
        
        # Agent phase
        if not candidate_cu.is_file():
            log("  Phase 1: AGENT")
            try:
                import os as _os
                _os.environ.setdefault("CODEX_HOME", r"<LOCAL_USER_HOME>\.codex")
                from lab.runtime.evaluators.remote_v100_campaign import build_agent_prompt
                from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
                BIN = Path(r"<PROJECT_ROOT>\runtimes\codex-0.154.0\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe")
                config = CodexConfig(codex_bin=str(BIN), cwd=str(ep_dir), client_name="aka_v100", client_title="aka-v100", client_version="0.5", config_overrides=("model_provider=openai",))
                cx = Codex(config)
                try:
                    prompt = build_agent_prompt(operator, "v100_sm70")
                    t = cx.thread_start(model="gpt-5.6-luna", model_provider="openai", cwd=str(ep_dir), sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all, ephemeral=True)
                    t.run(prompt, model="gpt-5.6-luna", effort="low", cwd=str(ep_dir), sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all)
                    log("  Agent complete")
                finally:
                    cx.close()
            except Exception as e:
                log(f"  Agent failed: {e}")
                write_json(ep_dir / "agent_error.json", {"timestamp": utcnow(), "error": str(e)})
                failed += 1
                continue
        
        if not candidate_cu.is_file():
            log("  No candidate, skipping")
            failed += 1
            continue
        
        # Phase 11.6: Contract marker validation (local, no GPU)
        contract_valid, contract_reason = validate_candidate_contract(ep_dir, operator)
        if not contract_valid:
            log(f"  Contract INVALID: {contract_reason}")
            decision_data = {
                "episode": episode_num, "operator": operator,
                "decision": "REJECT_CONTRACT",
                "compile_pass": False, "correctness_pass": False,
                "aggregate_score": 1.0, "geometric_mean_speedup": 1.0,
                "incumbent_score": get_incumbent_score_safe(operator),
                "reason": contract_reason,
                "evaluated_at": utcnow(),
                "score_type": "geometric_mean_speedup",
                "shapes": shapes,
            }
            write_json(ep_dir / "decision.json", decision_data)
            write_contract_rejection_lesson(operator, episode_num, contract_reason)
            failed += 1
            continue
        log(f"  Contract: {contract_reason}")

        # Validate hypothesis
        valid, hyp_errors = validate_hypothesis(ep_dir)
        if not valid:
            log(f"  Hypothesis INVALID: {hyp_errors}")
            write_hypothesis_validation_lesson(operator, episode_num, hyp_errors)
            failed += 1
            continue
        
        # Evaluate
        log("  Phase 2: EVALUATION")
        try:
            run_evaluation_for_episode(ep_dir, episode_num, operator, shapes, with_profile=False)
            decision = read_json(ep_dir / "decision.json", {})
            log(f"  Result: {decision.get('decision')} score={decision.get('aggregate_score')}")
            completed += 1
        except Exception as e:
            log(f"  Eval failed: {e}")
            failed += 1
    
    log(f"=== Done: {completed} ok, {failed} failed ===")
    return completed, failed


# ============================================================
# Task 5: Failure Injection Tests
# ============================================================

def test_failure_injection():
    """Phase 9: Run failure injection tests."""
    results = {}
    
    # Test 1: Hypothesis validation
    print("=== Test 1: Hypothesis Validation ===")
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    
    # Valid
    valid_hyp = {"strategy_tags": ["warp_shuffle_reduction"], "expected_effects": ["faster"], "risk": ["reg pressure"]}
    write_json(tmp / "hypothesis.json", valid_hyp)
    ok, errs = validate_hypothesis(tmp)
    results["valid_hypothesis"] = "PASS" if ok else f"FAIL: {errs}"
    print(f"  Valid hypothesis: {results['valid_hypothesis']}")
    
    # Missing fields
    bad_hyp = {"claim": "test"}
    write_json(tmp / "hypothesis.json", bad_hyp)
    ok, errs = validate_hypothesis(tmp)
    results["missing_fields"] = "PASS" if not ok else "FAIL (should reject)"
    print(f"  Missing fields:  {results['missing_fields']}")
    
    # Unknown strategy tag
    bad_hyp2 = {"strategy_tags": ["bad_tag"], "expected_effects": ["x"], "risk": ["y"]}
    write_json(tmp / "hypothesis.json", bad_hyp2)
    ok, errs = validate_hypothesis(tmp)
    results["unknown_tag"] = "PASS" if not ok else "FAIL (should reject)"
    print(f"  Unknown tag:     {results['unknown_tag']}")
    
    # Test 2: Lesson generation
    card = write_hypothesis_validation_lesson("test_op", 99, ["missing strategy_tags"])
    card_path = ROOT / "knowledge" / "environments" / "v100_sm70" / "lessons" / "test_op_rejected_episode_99.json"
    results["lesson_generated"] = "PASS" if card_path.is_file() else "FAIL"
    print(f"  Lesson generated: {results['lesson_generated']}")
    card_path.unlink()  # Cleanup
    
    # Test 3: Recommendations
    print("\n=== Test 2: Recommendations ===")
    recs = build_recommendations(operator, "v100_sm70")
    results["recommendations_count"] = f"PASS ({len(recs.get('recommendations',[]))} recs)" if recs.get("recommendations") else "FAIL"
    print(f"  Recommendations: {results['recommendations_count']}")
    if recs.get("recommendations"):
        top = recs["recommendations"][0]
        print(f"    Top: {top['name']} confidence={top['confidence']}")
        results["has_confidence"] = "PASS" if top.get("confidence") else "FAIL"
        results["has_apply_when"] = "PASS" if top.get("apply_when") else "FAIL"
    
    # Summary
    passed = sum(1 for v in results.values() if v.startswith("PASS"))
    total = len(results)
    print(f"\n=== Results: {passed}/{total} passed ===")
    return results

# ============================================================
# Phase 11.6: Contract validation helpers
# ============================================================

def load_operator_contract(operator):
    """Load structured contract for operator."""
    from lab.core.contract import OperatorContract, contract_from_legacy_metadata
    meta_path = ROOT / "operators" / operator / "metadata.json"
    if meta_path.is_file():
        meta = read_json(meta_path, {})
        if "contract_schema" in meta:
            return OperatorContract.from_dict(meta["contract_schema"])
        return contract_from_legacy_metadata(meta)
    return None


def validate_candidate_contract(ep_dir, operator):
    """Validate candidate.cu against operator contract.

    Checks:
    1. AKA_CONTRACT marker exists and matches
    2. hypothesis.json contract fields match (if present)

    Returns (ok, reason).
    """
    from lab.core.contract import (
        validate_candidate_contract_marker,
        validate_hypothesis_contract,
    )

    contract = load_operator_contract(operator)
    if contract is None:
        return True, "no contract to validate"

    candidate_cu = ep_dir / "candidate.cu"
    if not candidate_cu.is_file():
        return False, "candidate.cu not found"

    # Check AKA_CONTRACT marker
    ok, reason = validate_candidate_contract_marker(candidate_cu, contract)
    if not ok:
        return False, reason

    # Check hypothesis contract fields
    hyp_path = ep_dir / "hypothesis.json"
    if hyp_path.is_file():
        ok, reason = validate_hypothesis_contract(hyp_path, contract)
        if not ok:
            return False, f"hypothesis: {reason}"

    return True, f"contract {contract.contract_hash} matched"


def get_incumbent_score_safe(operator):
    """Get incumbent score safely, returning 1.0 if unknown."""
    try:
        from lab.runtime.evaluators.remote_v100_campaign import get_incumbent_score
        return get_incumbent_score(operator)
    except Exception:
        return 1.0


def write_contract_rejection_lesson(operator, episode_num, reason):
    """Record a contract rejection in knowledge."""
    env_knowledge = ROOT / "knowledge" / "environments" / "v100_sm70" / operator / "lessons"
    card = {
        "operator": operator,
        "episode": episode_num,
        "environment": "v100_sm70",
        "decision": "REJECT_CONTRACT",
        "failure_stage": "contract",
        "reason": reason,
        "reusable_rule": f"Contract validation failed: {reason}. Ensure candidate.cu has AKA_CONTRACT marker matching operator metadata.",
        "timestamp": utcnow(),
    }
    write_json(env_knowledge / f"episode_{episode_num}.json", card)