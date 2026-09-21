"""Run the mechanical post-generation pipeline for one aka-local episode.

This entry point never imports or invokes Codex. It validates an existing
candidate, invokes the official Atrex-Bench compile/correctness evaluator,
then runs the local same-process ABBA benchmark and applies the existing
strict-positive-improvement policy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ATREX_ROOT = Path(r"<LOCAL_USER_HOME>\projects\atrex-bench")
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
INCUMBENT = ROOT / "ops" / "swiglu_forward_v2b" / "kernel.py"
REFERENCE = ATREX_ROOT / "local_ops" / "swiglu_forward_v2b"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: dict[str, Any], *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise FileExistsError(f"refusing to overwrite existing terminal result: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def transition(state_path: Path, state: str, **extra: Any) -> dict[str, Any]:
    current = {"state": state, "updated_at": now(), **extra}
    save_json(state_path, current)
    return current


def terminal(result_path: Path, state_path: Path, result: dict[str, Any]) -> int:
    save_json(result_path, result, refuse_existing=True)
    transition(state_path, result["state"], failure_reason=result.get("failure_reason"))
    return 0 if result["state"] == "ACCEPTED" else 1


def find_eval_result(root: Path) -> Path | None:
    candidates = sorted(root.rglob("eval_result.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def all_stage_passed(payload: dict[str, Any], stage: str) -> bool:
    block = payload.get("passed", {}).get(stage)
    return isinstance(block, dict) and bool(block) and all(
        isinstance(row, dict) and row.get("status") == "passed" for row in block.values()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Mechanical aka-local V3 episode evaluator")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=INCUMBENT)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    episode = args.episode.resolve()
    baseline = args.baseline.resolve()
    reference = args.reference_dir.resolve()
    output = (args.output or episode / "mechanical").resolve()
    state_path = output / "state.json"
    result_path = output / "decision.json"
    output.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        print(f"TERMINAL_RESULT_EXISTS={result_path}")
        return 2

    transition(state_path, "CREATED", model_calls=0)
    candidate = episode / "kernel.py"
    required = [
        candidate,
        episode / "swiglu_kernel_v3.cu",
        baseline,
        reference / "reference.py",
        reference / "input.py",
        reference / "shapes.json",
        reference / "metadata.json",
        reference / "roofline.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        reference_issue = any(str(reference) in item for item in missing)
        failure_state = "REFERENCE_INVALID" if reference_issue else "CANDIDATE_MISSING"
        return terminal(result_path, state_path, {
            "schema_version": 1, "episode": 3, "candidate": "V3", "baseline": "V2b",
            "model_calls": 0, "state": failure_state,
            "candidate_status": "missing" if not candidate.is_file() else "present",
            "compile": None, "correctness": None, "benchmark": None, "abba": None,
            "decision": "not_applicable", "failure_reason": "missing files: " + "; ".join(missing),
        })

    transition(state_path, "CANDIDATE_READY", candidate=str(candidate))
    eval_root = output / "atrex_eval"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ATREX_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cuda_path = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"))
    env["PATH"] = os.pathsep.join(
        [str(PYTHON.parent), str(cuda_path / "bin"), env.get("PATH", "")]
    )
    command = [
        str(PYTHON), "-m", "atrex_bench.cli.run_eval",
        "--input", str(candidate), "--reference-dir", str(reference), "--output", str(eval_root),
        "--correctness-only", "--candidate-timeout-s", "0", "--compile-timeout-s", "0",
        "--atol", "0.01", "--rtol", "0.05", "--num-correctness-cases", "1",
    ]
    transition(state_path, "COMPILE_RUNNING", command=command)
    completed = subprocess.run(
        command,
        cwd=str(ATREX_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    eval_result = find_eval_result(eval_root)
    if eval_result is None:
        return terminal(result_path, state_path, {
            "schema_version": 1, "episode": 3, "candidate": "V3", "baseline": "V2b",
            "model_calls": 0, "state": "RESULT_PARSE_FAILED", "candidate_status": "present",
            "compile": None, "correctness": None, "benchmark": None, "abba": None,
            "decision": "not_applicable", "failure_reason": "official evaluator emitted no eval_result.json",
            "evaluator_returncode": completed.returncode,
        })
    payload = json.loads(eval_result.read_text(encoding="utf-8"))
    compile_pass = all_stage_passed(payload, "compile")
    if not compile_pass:
        transition(state_path, "COMPILE_FAILED", evaluator_result=str(eval_result))
        return terminal(result_path, state_path, {
            "schema_version": 1, "episode": 3, "candidate": "V3", "baseline": "V2b",
            "model_calls": 0, "state": "COMPILE_FAILED", "candidate_status": "present",
            "compile": payload.get("passed", {}).get("compile"), "correctness": None,
            "benchmark": None, "abba": None, "decision": "REJECT_COMPILE",
            "failure_reason": payload.get("error") or "compile stage failed",
            "evaluator_result": str(eval_result),
        })
    transition(state_path, "COMPILE_PASS", evaluator_result=str(eval_result))
    correctness_pass = all_stage_passed(payload, "correctness")
    if not correctness_pass:
        transition(state_path, "CORRECTNESS_FAILED", evaluator_result=str(eval_result))
        return terminal(result_path, state_path, {
            "schema_version": 1, "episode": 3, "candidate": "V3", "baseline": "V2b",
            "model_calls": 0, "state": "CORRECTNESS_FAILED", "candidate_status": "present",
            "compile": payload.get("passed", {}).get("compile"),
            "correctness": payload.get("correctness"), "benchmark": None, "abba": None,
            "decision": "REJECT_CORRECTNESS", "failure_reason": payload.get("error") or "correctness failed",
            "evaluator_result": str(eval_result),
        })
    transition(state_path, "CORRECTNESS_PASS", evaluator_result=str(eval_result))

    transition(state_path, "BENCHMARK_RUNNING")
    abba_path = output / "abba.json"
    abba_command = [str(PYTHON), str(ROOT / "benchmarks" / "swiglu_abba.py"), "--incumbent", str(baseline), "--candidate", str(candidate), "--output", str(abba_path), "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
    abba = subprocess.run(
        abba_command,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if abba.returncode != 0 or not abba_path.is_file():
        return terminal(result_path, state_path, {
            "schema_version": 1, "episode": 3, "candidate": "V3", "baseline": "V2b",
            "model_calls": 0, "state": "BENCHMARK_FAILED", "candidate_status": "present",
            "compile": payload.get("passed", {}).get("compile"), "correctness": payload.get("correctness"),
            "benchmark": None, "abba": None, "decision": "REJECT_PERFORMANCE",
            "failure_reason": abba.stderr[-4000:] or "ABBA benchmark failed",
            "evaluator_result": str(eval_result),
        })
    abba_payload = json.loads(abba_path.read_text(encoding="utf-8"))
    speedup = abba_payload.get("arithmetic_mean_speedup")
    if not isinstance(speedup, (int, float)):
        return terminal(result_path, state_path, {
            "schema_version": 1, "episode": 3, "candidate": "V3", "baseline": "V2b",
            "model_calls": 0, "state": "RESULT_PARSE_FAILED", "candidate_status": "present",
            "compile": payload.get("passed", {}).get("compile"), "correctness": payload.get("correctness"),
            "benchmark": None, "abba": abba_payload, "decision": "not_applicable",
            "failure_reason": "ABBA result missing arithmetic_mean_speedup",
            "evaluator_result": str(eval_result),
        })
    transition(state_path, "BENCHMARK_PASS", abba=str(abba_path))
    accepted = float(speedup) > 1.0
    state = "ACCEPTED" if accepted else "REJECTED"
    return terminal(result_path, state_path, {
        "schema_version": 1, "episode": 3, "candidate": "V3", "baseline": "V2b",
        "model_calls": 0, "candidate_status": "present", "state": state,
        "compile": payload.get("passed", {}).get("compile"), "correctness": payload.get("correctness"),
        "benchmark": {"protocol": "same_process_ABBA", "status": "passed"}, "abba": abba_payload,
        "decision": "ACCEPT" if accepted else "REJECT_PERFORMANCE",
        "failure_reason": None,
        "decision_policy": {"source": "long_horizon.verifier strict improvement", "min_improvement_pct": 0.0, "metric": "arithmetic_mean_speedup_vs_incumbent", "strict": "> 1.0"},
        "evaluator_result": str(eval_result),
    })


if __name__ == "__main__":
    raise SystemExit(main())
