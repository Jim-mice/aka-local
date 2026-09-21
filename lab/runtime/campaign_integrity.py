'''Campaign integrity module for AKA-local continuous runner.

Phase 7-A: Provides filesystem-first episode numbering, promotion lineage
tracking, episode manifests with SHA-256 hashes, automatic knowledge
precipitation (experience/lesson cards), state validation, and startup
diagnostics.

Do NOT modify CUDA logic, Agent prompts, evaluator, or Supervisor thresholds.
'''

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"<PROJECT_ROOT>")
ATREX = Path(r"<LOCAL_USER_HOME>\projects\atrex-bench")
RMS_REF_DIR = ATREX / "data" / "rms_norm"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    '''Return SHA-256 hex digest of a file.'''
    return hashlib.sha256(path.read_bytes()).hexdigest()


def warn(msg: str) -> None:
    print(f'[STATE WARNING] {msg}', flush=True)


# ============================================================
# Task 1: Episode numbering safety (filesystem-first)
# ============================================================

def find_next_episode(operator: str) -> int:
    '''Scan campaign directory and return next available episode number.

    Priority: filesystem > continuous_state.json.
    Never overwrite existing episodes.
    '''
    campaign_dir = ROOT / "campaigns" / operator
    existing = set()
    if campaign_dir.is_dir():
        for d in campaign_dir.iterdir():
            m = re.match(r'^episode_(\d+)$', d.name)
            if m and d.is_dir():
                existing.add(int(m.group(1)))
    return max(existing) + 1 if existing else 1


def check_and_correct_next_episode(state: dict, operator: str) -> tuple[int, bool]:
    '''Compare state next_episode against filesystem and correct if needed.

    Returns (corrected_next_episode, was_corrected).
    '''
    fs_next = find_next_episode(operator)
    sn = int(state.get("next_episode", 1))

    if sn < fs_next:
        print("[STATE WARNING]", flush=True)
        print(f"  next_episode corrected:", flush=True)
        print(f"  old = {sn}", flush=True)
        print(f"  new = {fs_next}", flush=True)
        return fs_next, True

    return sn, False


# ============================================================
# Task 2: Promotion lineage
# ============================================================

def append_lineage(operator: str, entry: dict) -> None:
    '''Append a promotion entry to lineage.jsonl (append-only, never overwrite).'''
    lp = ROOT / "campaigns" / operator / "lineage.jsonl"
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def count_lineage_entries(operator: str) -> int:
    '''Count entries in lineage.jsonl.'''
    lp = ROOT / "campaigns" / operator / "lineage.jsonl"
    if not lp.is_file():
        return 0
    try:
        return sum(1 for _ in open(str(lp), encoding="utf-8"))
    except Exception:
        return 0


# ============================================================
# Task 3: Episode manifest with hashes
# ============================================================

def write_episode_manifest(ep: Path, operator: str,
                           candidate_py: Path, baseline_py: Path) -> dict:
    '''Write episode_manifest.json with SHA-256 hashes for reproducibility.'''
    ref_py = RMS_REF_DIR / "reference.py"
    manifest = {
        "episode": int(ep.name.split("_")[-1]),
        "operator": operator,
        "candidate_sha256": sha256_file(candidate_py) if candidate_py.is_file() else None,
        "baseline_sha256": sha256_file(baseline_py) if baseline_py.is_file() else None,
        "reference_sha256": sha256_file(ref_py) if ref_py.is_file() else None,
        "created_time": utcnow(),
    }
    write_json(ep / "episode_manifest.json", manifest)
    return manifest


# ============================================================
# Task 4: Auto-generate experience / lesson cards
# ============================================================

def _extract_hypothesis_info(ep: Path) -> dict:
    hp = ep / "hypothesis.json"
    if not hp.is_file():
        return {}
    try:
        h = json.loads(hp.read_text(encoding="utf-8-sig"))
        return {
            "direction_id": h.get("direction_id", ""),
            "summary": h.get("hypothesis", h.get("description", "")),
        }
    except Exception:
        return {}


def write_experience_card(operator: str, ep: Path,
                          speedup: float, geo_speedup: float,
                          new_incumbent: str) -> None:
    '''Write experience card to knowledge/experience/ on PROMOTE.'''
    en = int(ep.name.split("_")[-1])
    hi = _extract_hypothesis_info(ep)
    am = ep / "AGENT.md"
    at = am.read_text(encoding="utf-8")[:2000] if am.is_file() else ""

    patterns = []
    for kw in ["warp", "shared memory", "reduction", "register", "tiling",
               "vectorize", "prefetch", "unroll", "pipeline", "async copy"]:
        if kw.lower() in at.lower():
            patterns.append(kw)

    card = {
        "operator": operator,
        "episode": en,
        "hypothesis": hi,
        "result": {
            "decision": "PROMOTE",
            "speedup": round(speedup, 6),
            "geometric_speedup": round(geo_speedup, 6),
        },
        "lesson": (
            f"Episode {en}: {speedup:.4f}x speedup "
            f"(geo {geo_speedup:.4f}x) on sm_120 "
            f"via {hi.get('direction_id', 'unknown')}."
        ),
        "reusable_pattern": patterns,
    }
    out = ROOT / "knowledge" / "experience" / f"{operator}_episode_{en}.json"
    write_json(out, card)
    print(f"[DEBUG] experience card: {out.name}", flush=True)


def write_lesson_card(operator: str, ep: Path, eval_result: dict) -> None:
    '''Write lesson card to knowledge/lessons/ on REJECT.'''
    en = int(ep.name.split("_")[-1])
    sv = eval_result.get("state", "UNKNOWN")

    rm = {
        "COMPILE_FAILED": "compile failure",
        "CORRECTNESS_FAILED": "correctness failure",
        "BENCHMARK_FAILED": "performance regression",
        "REJECTED": "performance regression",
    }

    # Phase 7-C Task 5: extract compiler error summary
    compiler_error = ""
    cs = eval_result.get("compiler_error_summary", "")
    if cs:
        compiler_error = cs
    elif sv == "COMPILE_FAILED":
        # Fallback: try to extract from mechanical results
        try:
            mech = ep / "mechanical" / "atrex_eval"
            if mech.is_dir():
                results = sorted(mech.rglob("eval_result.json"),
                                 key=lambda p: p.stat().st_mtime, reverse=True)
                if results:
                    payload = json.loads(results[0].read_text(encoding="utf-8"))
                    err = payload.get("error", "")
                    if err:
                        compiler_error = err[:500]
        except Exception:
            pass

    lesson = {
        "operator": operator,
        "episode": en,
        "decision": "REJECT",
        "reason": eval_result.get("failure_reason", rm.get(sv, sv)),
        "compile_pass": eval_result.get("state") not in ("COMPILE_FAILED",),
        "correctness_pass": eval_result.get("state")
            not in ("COMPILE_FAILED", "CORRECTNESS_FAILED"),
        "speedup": (
            eval_result.get("abba", {}).get("arithmetic_mean_speedup")
            if isinstance(eval_result.get("abba"), dict) else None
        ),
        "failure_stage": sv,
        "compiler_error_summary": compiler_error[:500] if compiler_error else "",
        "timestamp": utcnow(),
    }
    out = ROOT / "knowledge" / "lessons" / f"{operator}_rejected_episode_{en}.json"
    write_json(out, lesson)
    print(f"[DEBUG] lesson card: {out.name}", flush=True)


# ============================================================
# Task 5: State validation
# ============================================================

def validate_state(state: dict, operator: str) -> dict:
    '''Validate and auto-repair continuous_state.json at startup.

    Checks:
      1. next_episode >= filesystem_max_episode + 1
      2. current_incumbent must exist at ops/<incumbent>/candidate.py
      3. PROMOTED state must have episode/decision.json
      4. REJECTED state must not modify incumbent (enforced by design)
    '''
    fixed = dict(state)
    issues = []

    # --- Check 1: next_episode ---
    fs_next = find_next_episode(operator)
    sn = int(fixed.get("next_episode", 1))
    if sn < fs_next:
        print("[STATE WARNING]", flush=True)
        print(f"  next_episode corrected:", flush=True)
        print(f"  old = {sn}", flush=True)
        print(f"  new = {fs_next}", flush=True)
        fixed["next_episode"] = fs_next

    # --- Check 2: current_incumbent must exist ---
    incumbent = fixed.get("current_incumbent", "")
    ipy = ROOT / "ops" / incumbent / "candidate.py"
    if not ipy.is_file():
        ov = sorted(
            [d for d in (ROOT / "ops").iterdir()
             if d.is_dir() and re.match(rf"{operator}_v\d+", d.name)
             and (d / "candidate.py").is_file()],
            key=lambda d: int(re.match(r".*_v(\d+)", d.name).group(1)),
            reverse=True,
        )
        if ov:
            ni = ov[0].name
            print("[STATE WARNING]", flush=True)
            print(f"  incumbent missing: {incumbent} -> {ni}", flush=True)
            fixed["current_incumbent"] = ni
        else:
            print("[STATE WARNING]", flush=True)
            print(f"  incumbent missing: {incumbent}, no fallback", flush=True)

    # --- Check 3: PROMOTED must have decision.json ---
    if fixed.get("current_state") == "PROMOTED":
        pe = ROOT / "campaigns" / operator / f"episode_{fixed['next_episode'] - 1}"
        if not (pe / "decision.json").is_file():
            print("[STATE WARNING]", flush=True)
            print("  state=PROMOTED but decision.json missing", flush=True)
            fixed["current_state"] = "CANDIDATE_READY"

    # --- Check 4: REJECTED never modifies incumbent ---
    # This is enforced by handle_decision() design: REJECTED only
    # increments next_episode, never changes current_incumbent.
    # Validate that this invariant holds in the state file.
    if fixed.get("current_state") == "REJECTED":
        # Verify the incumbent wasn't accidentally changed
        # (no action needed; just document the invariant is intact)
        pass

    return fixed


# ============================================================
# Task 6: Campaign dashboard / startup diagnostics
# ============================================================

def print_banner(state: dict, operator: str) -> None:
    '''Print startup diagnostics banner.'''
    cd = ROOT / "campaigns" / operator
    eps = sorted(
        [d for d in cd.iterdir()
         if re.match(r'^episode_\d+$', d.name) and d.is_dir()],
        key=lambda d: int(re.match(r"episode_(\d+)", d.name).group(1)),
    )
    le = eps[-1].name if eps else "none"
    lc = count_lineage_entries(operator)
    ec = list((ROOT / "knowledge" / "experience").glob(f"{operator}_episode_*.json"))
    lsc = list((ROOT / "knowledge" / "lessons").glob(f"{operator}_rejected_*.json"))

    print("=" * 60, flush=True)
    print("  AKA Campaign Integrity", flush=True)
    print("=" * 60, flush=True)
    print(f"  operator           = {operator}", flush=True)
    print(f"  current incumbent  = {state.get('current_incumbent', '?')}", flush=True)
    print(f"  latest episode     = {le}", flush=True)
    print(f"  next episode       = {state.get('next_episode', '?')}", flush=True)
    print(f"  lineage entries    = {lc}", flush=True)
    print(f"  experience cards   = {len(ec)}", flush=True)
    print(f"  lesson cards       = {len(lsc)}", flush=True)
    print(f"  current state      = {state.get('current_state', '?')}", flush=True)
    print("=" * 60, flush=True)

