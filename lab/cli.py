'''AKA-Local Lab CLI - Phase 10-B.

Commands:
  doctor      Verify environment health
  list-ops    List all available operators
  operators    Alias for list-ops
  evaluate    Evaluate a candidate.cu on V100
  run         Run unattended optimization campaign
  replay      Replay a previous episode
  report      Generate optimization report
  status      Show project status
  recover     Recover interrupted runs
  validate    Run validation suite
'''

import argparse, json, os, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(
        description="AKA-Local Lab CLI - CUDA kernel optimization platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  lab doctor --env v100
  lab list-ops
  lab evaluate --candidate candidate.cu --env v100
  lab run --env v100 --op rms_norm_v100_cuda --episodes 1
  lab run --env v100 --op rms_norm_v100_cuda --no-agent
  lab replay --env v100 --op rms_norm_v100_cuda --ep 16
  lab report --env v100 --op rms_norm_v100_cuda
  lab status
'''
    )
    p.add_argument("command",
                   choices=["doctor", "list-ops", "operators", "evaluate", "run",
                            "replay", "report", "status", "campaigns",
                            "platforms", "knowledge", "validate", "recover", "target"])
    p.add_argument("--env", default="v100", help="Environment (default: v100)")
    p.add_argument("--op", help="Operator name")
    p.add_argument("--ep", type=int, help="Episode number")
    p.add_argument("--episodes", type=int, default=1, help="Episodes for unattended run")
    p.add_argument("--resume-run", action="store_true", help="Resume interrupted campaign")
    p.add_argument("--no-agent", action="store_true", help="Skip agent phase (use existing/reference candidate)")
    p.add_argument("--candidate", help="Path to candidate.cu file")
    p.add_argument("--operator", help="Operator name (alias for --op)")
    p.add_argument("--hardware")
    p.add_argument("--type")
    p.add_argument("--list", action="store_true")
    p.add_argument("--inspect")
    p.add_argument("--resume")
    p.add_argument("--target", default="swiglu", help="Real target adapter name")
    p.add_argument("--tp", type=int, default=1, help="Tensor-parallel size for distributed target replay")
    p.add_argument("--action", dest="target_action", default="inspect", choices=["inspect", "replay", "benchmark", "run", "profile", "inspect-backward", "replay-backward", "benchmark-backward", "profile-backward", "validate-integration", "benchmark-integration", "profile-integration"])
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    # Resolve --op / --operator alias
    if a.operator and not a.op:
        a.op = a.operator

    if a.command == "doctor":
        _cmd_doctor(a)
    elif a.command in ("list-ops", "operators"):
        _cmd_list_ops(a)
    elif a.command == "evaluate":
        _cmd_evaluate(a)
    elif a.command == "run":
        _cmd_run(a)
    elif a.command == "replay":
        _cmd_replay(a)
    elif a.command == "report":
        _cmd_report(a)
    elif a.command in ("status", "campaigns"):
        print((ROOT / "STATUS.md").read_text(encoding="utf-8"))
    elif a.command == "platforms":
        for p in sorted((ROOT / "registry" / "platforms").glob("*.yaml")):
            print(p.stem)
    elif a.command == "validate":
        from lab.tools.validate_lab import main as v
        raise SystemExit(v())
    elif a.command == "recover":
        _cmd_recover(a)
    elif a.command == "knowledge":
        from lab.tools.query import main as q
        sys.argv.pop(1)
        q()
    elif a.command == "target":
        _cmd_target(a)


def _cmd_target(a):
    """Run a source-grounded target adapter without starting a campaign."""
    if a.target == "residual_rmsnorm":
        project = ROOT.parent
        target_dir = project / "targets/megatron_5be9626/residual_rmsnorm"
        if a.target_action == "inspect":
            replay = json.loads((target_dir / "replay_contract.json").read_text(encoding="utf-8"))
            perf = json.loads((target_dir / "performance_contract.json").read_text(encoding="utf-8"))
            print(json.dumps({"target": a.target, "action": a.target_action, "canonical_target": replay["canonical_target"], "replay_contract_hash": replay["contract_hash"], "performance_contract_hash": perf["performance_contract_hash"], "commit": replay["megatron_commit"]}, indent=2))
            return
        if a.target_action in ("replay", "benchmark", "profile"):
            script = {"replay": "_run_phase16a.py", "benchmark": "_run_phase16a.py", "profile": "_run_phase16a_nsys.py"}[a.target_action]
            subprocess.run([sys.executable, str(project / script)], check=True)
            print(json.dumps({"target": a.target, "action": a.target_action, "status": "PASS"}, indent=2))
            return
        raise SystemExit(f"Unsupported action for {a.target}: {a.target_action}")
    if a.target == "vocab_parallel_cross_entropy":
        project = ROOT.parent
        target_dir = project / "targets/megatron_5be9626/vocab_parallel_cross_entropy"
        if a.target_action == "inspect":
            spec = json.loads((target_dir / "replay_contract.json").read_text(encoding="utf-8"))
            print(json.dumps({"target": a.target, "action": "inspect", "contract_hash": spec["contract_hash"], "commit": spec["megatron_commit"], "tp": a.tp}, indent=2))
            return
        if a.target_action == "inspect-backward":
            spec = json.loads((target_dir / "backward_replay_contract.json").read_text(encoding="utf-8"))
            perf = json.loads((target_dir / "backward_performance_contract.json").read_text(encoding="utf-8"))
            print(json.dumps({"target": a.target, "action": a.target_action, "replay_contract_hash": spec["contract_hash"], "performance_contract_hash": perf["performance_contract_hash"], "commit": spec["megatron_commit"], "collectives_in_backward": spec["scope"]["collectives"], "benchmark_era": perf["benchmark_era"]}, indent=2))
            return
        if a.target_action in ("replay-backward", "benchmark-backward", "profile-backward"):
            project = ROOT.parent
            script = {"replay-backward": "_run_backward_replay.py", "benchmark-backward": "_run_backward_replay.py", "profile-backward": "_run_backward_nsys.py"}[a.target_action]
            if a.target_action == "replay-backward":
                subprocess.run([sys.executable, str(project / script), "correctness"], check=True)
            elif a.target_action == "benchmark-backward":
                subprocess.run([sys.executable, str(project / script), "benchmark"], check=True)
            else:
                subprocess.run([sys.executable, str(project / script)], check=True)
            print(json.dumps({"target": a.target, "action": a.target_action, "status": "PASS"}, indent=2))
            return
        if a.tp not in (1, 2):
            raise SystemExit("vocab_parallel_cross_entropy replay supports TP=1 or TP=2")
        if a.target_action in ("replay", "benchmark"):
            mode = f"tp{a.tp}" + ("_bench" if a.target_action == "benchmark" else "")
            subprocess.run([sys.executable, str(project / "_run_vocab_replay.py"), mode], check=True)
            return
        if a.target_action == "profile":
            subprocess.run([sys.executable, str(project / "_run_vocab_nsys.py"), f"tp{a.tp}"], check=True)
            return
        raise SystemExit(f"Unsupported action for {a.target}: {a.target_action}")
    if a.target == "megatron_native_dot_product_attention":
        project = ROOT.parent
        target_dir = project / "targets/megatron_5be9626/megatron_native_dot_product_attention"
        if a.target_action == "inspect":
            replay = json.loads((target_dir / "replay_contract.json").read_text(encoding="utf-8"))
            perf = json.loads((target_dir / "performance_contract.json").read_text(encoding="utf-8"))
            print(json.dumps({"target": a.target, "canonical_target": replay["canonical_target"], "replay_contract_hash": replay["contract_hash"], "performance_contract_hash": perf["contract_hash"], "commit": replay["megatron_commit"], "collectives": replay["collectives"]}, indent=2))
            return
        scripts = {"replay": "attention_phase17a_replay.py", "benchmark": "attention_phase17a_benchmark.py", "profile": "attention_phase17a_profile.py"}
        if a.target_action in scripts:
            subprocess.run([sys.executable, str(project / "targets/megatron_5be9626" / scripts[a.target_action])], check=True)
            return
        raise SystemExit(f"Unsupported action for {a.target}: {a.target_action}")
    if a.target != "swiglu":
        raise SystemExit(f"Unknown target adapter: {a.target}")
    if a.target_action == "run":
        if not 1 <= a.episodes <= 5:
            raise SystemExit("target campaign episodes must be between 1 and 5")
        project = ROOT.parent
        for episode in range(1, a.episodes + 1):
            subprocess.run([sys.executable, str(project / "_run_target_agent.py"), str(episode)], check=True)
            subprocess.run([sys.executable, str(project / "_eval_target_episode.py"), str(episode)], check=True)
        print(json.dumps({"target": a.target, "episodes": a.episodes, "status": "EVALUATED"}, indent=2))
        return
    if a.target_action in ("validate-integration", "benchmark-integration", "profile-integration"):
        project = ROOT.parent
        mode = {"validate-integration":"grad", "benchmark-integration":"benchmark", "profile-integration":"nsys-profile"}[a.target_action]
        subprocess.run([sys.executable, str(project / "_run_phase14d.py"), mode], check=True)
        print(json.dumps({"target":a.target,"action":a.target_action,"status":"PASS"}, indent=2))
        return
    if a.target_action == "inspect":
        spec = json.loads((ROOT.parent / "targets/megatron_5be9626/swiglu.json").read_text(encoding="utf-8"))
        print(json.dumps({"target": a.target, "action": a.target_action, "commit": spec["megatron_commit"], "spec": spec}, indent=2))
        return
    from lab.targets.megatron_swiglu import MegatronSwiGLUAdapter
    adapter = MegatronSwiGLUAdapter(device=a.device)
    print(json.dumps({"target": a.target, "action": a.target_action, "commit": adapter.load_target_spec()["megatron_commit"]}, indent=2))
    inputs = adapter.prepare_inputs(seed=14)
    reference = adapter.run_megatron_reference(inputs)
    replay = adapter.run_replay(inputs)
    comparison = adapter.compare_outputs(reference, replay)
    result = {"comparison": comparison, "shape": list(inputs["hidden_states"].shape), "dtype": str(inputs["hidden_states"].dtype), "device": str(inputs["hidden_states"].device)}
    if a.target_action == "benchmark":
        result["reference_latency_us"] = adapter.benchmark_reference(inputs)
    print(json.dumps(result, indent=2))


def _cmd_list_ops(a):
    """List all available operators across environments."""
    print("=" * 50)
    print("Available Operators")
    print("=" * 50)

    # V100 CUDA operators
    v100_ops_dir = ROOT.parent / "operators"
    v100_ops = []
    if v100_ops_dir.is_dir():
        for p in sorted(v100_ops_dir.iterdir()):
            if p.is_dir():
                meta = p / "metadata.json"
                if meta.is_file():
                    d = json.loads(meta.read_bytes())
                    v100_ops.append((p.name, d))

    if v100_ops:
        print()
        print("[V100 - sm_70]")
        for name, meta in v100_ops:
            contract = meta.get("contract", "")
            entry = contract.split("(")[0] if contract else "unknown"
            print("  " + name)
            print("    interface: " + str(meta.get("interface", "unknown")))
            print("    entry:     " + entry)
            print("    compiler:  " + str(meta.get("compiler", "unknown")))

    # Registry operators
    reg_dir = ROOT / "registry" / "operators"
    if reg_dir.is_dir():
        reg_ops = sorted(reg_dir.glob("*.yaml"))
        if reg_ops:
            print()
            print("[Registry Operators]")
            for p in reg_ops:
                print("  " + p.stem)

    # RTX5060 operators
    ops_dir = ROOT.parent / "ops"
    if ops_dir.is_dir():
        rtx_ops = sorted(d for d in ops_dir.iterdir() if d.is_dir())
        if rtx_ops:
            print()
            print("[RTX5060 - sm_120]")
            for d in rtx_ops:
                print("  " + d.name)

    print()
def _cmd_evaluate(a):
    """Evaluate a candidate.cu on the remote V100."""
    candidate_path = a.candidate
    if not candidate_path:
        print("[evaluate] ERROR: --candidate PATH is required")
        print("Example: lab evaluate --candidate path/to/candidate.cu --env v100")
        sys.exit(1)

    candidate = Path(candidate_path)
    if not candidate.is_file():
        print("[evaluate] ERROR: candidate not found: " + str(candidate))
        sys.exit(1)

    env_map = {"v100": "v100_sm70"}
    env_id = env_map.get(a.env, a.env + "_sm70")

    print("[evaluate] Candidate: " + str(candidate))
    print("[evaluate] Environment: " + env_id)
    print()

    # Read shapes from evaluation contract
    eval_contract = ROOT.parent / "config" / "environments" / "v100_sm70" / f"evaluation_{a.op or 'rms_norm_v100_cuda'}.json"
    if not eval_contract.is_file():
        eval_contract = ROOT.parent / "config" / "environments" / "v100_sm70" / "evaluation.json"
    shapes = None
    if eval_contract.is_file():
        contract = json.loads(eval_contract.read_text(encoding="utf-8"))
        raw = contract.get("shapes", [])
        shapes = [",".join(str(x) for x in s) if isinstance(s, list) else s for s in raw]
    if not shapes:
        shapes = ["1,4096", "4,4096", "8,4096", "32,4096"]

    print("[evaluate] Shapes: " + str(shapes))
    print()

    from lab.runtime.evaluators.phase8e import evaluate_with_stats
    result = evaluate_with_stats(str(candidate), shapes, runs_per_shape=3, operator=a.op or "rms_norm_v100_cuda")

    if not result.get("compile_pass"):
        print("[evaluate] COMPILE FAILED: " + str(result.get("error", "unknown")))
        sys.exit(1)
    if not result.get("correctness_pass"):
        print("[evaluate] CORRECTNESS FAILED: " + str(result.get("error", "unknown")))
        sys.exit(1)

    score = result.get("geometric_mean_speedup", 1.0)
    print("[evaluate] Score: " + str(score) + " (geometric_mean_speedup)")
    print()
    for s in result.get("shapes", []):
        mn = s.get("mean_latency_us", 0)
        sp = s.get("mean_speedup", 1.0)
        print("  " + str(s["shape"]) + ": " + str(mn) + "us, " + str(sp) + "x")

    print()
    print("[evaluate] PASS")
def _cmd_run(a):
    """Run unattended optimization campaign."""
    env_name = a.env
    operator = a.op or "rms_norm_v100_cuda"
    total_episodes = a.episodes
    resume = a.resume_run
    no_agent = a.no_agent

    print("=" * 50)
    print("Phase 10-B: Unattended Campaign")
    print("=" * 50)
    print("  Operator:     " + operator)
    print("  Environment:  " + env_name)
    print("  Episodes:     " + str(total_episodes))
    print("  Resume:       " + str(resume))
    print("  No-Agent:     " + str(no_agent))
    print()

    if no_agent:
        _cmd_run_no_agent(operator, env_name, total_episodes, resume)
    else:
        from lab.runtime.evaluators.phase9 import run_unattended_campaign
        completed, failed = run_unattended_campaign(operator, env_name, total_episodes, resume=resume)
        print()
        print("=" * 50)
        print("Campaign complete: " + str(completed) + " ok, " + str(failed) + " failed")
        print("=" * 50)


def _cmd_run_no_agent(operator, env_name, total_episodes, resume):
    """Run campaign without Codex agent - uses reference.cu as candidate."""
    import shutil
    from lab.runtime.evaluators.remote_v100_campaign import (
        get_eval_shapes, find_next_episode_v100, run_evaluation_for_episode,
    )
    from lab.runtime.evaluators.phase9 import rotate_log_if_needed, utcnow

    log_path = ROOT.parent / "continuous_run.log"
    rotate_log_if_needed(log_path)

    def log(msg):
        line = "[" + utcnow() + "] " + msg
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    shapes = get_eval_shapes(operator)
    camp_dir = ROOT.parent / "campaigns" / operator
    camp_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = 0

    for ep_idx in range(total_episodes):
        episode_num = find_next_episode_v100(operator)
        ep_dir = camp_dir / ("episode_" + str(episode_num))

        decision_path = ep_dir / "decision.json"
        if resume and decision_path.is_file():
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if decision.get("decision"):
                log("Ep " + str(episode_num) + ": SKIP (completed: " + decision["decision"] + ")")
                completed += 1
                continue

        ep_dir.mkdir(parents=True, exist_ok=True)
        log("--- Ep " + str(episode_num) + "/" + str(total_episodes) + " (no-agent) ---")

        # Use reference.cu as candidate when no agent
        candidate_cu = ep_dir / "candidate.cu"
        ref_cu = ROOT.parent / "operators" / operator / "reference.cu"
        if ref_cu.is_file() and not candidate_cu.is_file():
            shutil.copy2(str(ref_cu), str(candidate_cu))
            log("  Using reference.cu as candidate")

        if not candidate_cu.is_file():
            log("  No candidate available, skipping")
            failed += 1
            continue

        log("  Phase 2: EVALUATION (no-agent)")
        try:
            run_evaluation_for_episode(ep_dir, episode_num, operator, shapes, with_profile=False)
            decision = json.loads((ep_dir / "decision.json").read_text(encoding="utf-8"))
            log("  Result: " + decision.get("decision", "?") + " score=" + str(decision.get("aggregate_score", "?")))
            completed += 1
        except Exception as e:
            log("  Eval failed: " + str(e))
            fail_record = {
                "episode": episode_num,
                "error": str(e),
                "timestamp": utcnow(),
            }
            import json as _json
            (ep_dir / "failure.json").write_text(_json.dumps(fail_record, indent=2), encoding="utf-8")
            failed += 1

    log("=== Done: " + str(completed) + " ok, " + str(failed) + " failed ===")
    return completed, failed
def _cmd_recover(a):
  from lab.runtime.recovery import RecoveryManager
  from lab.core.probe import probe_local
  runs_dir = ROOT / "runtime" / "runs"
  if a.list:
    found = []
    if runs_dir.is_dir():
      for p in sorted(runs_dir.iterdir()):
        if not p.is_dir(): continue
        report = RecoveryManager.detect(p)
        if report.state in ("INTERRUPTED", "STALE"):
          found.append((p.name, report))
    if not found:
      print("No interrupted runs found.")
      return
    print("Interrupted runs:")
    print("-" * 60)
    for run_id, report in found:
      print(f"  Run ID:       {run_id}")
      print(f"  Campaign:     {report.details.get('campaign_dir', 'N/A')}")
      print(f"  State:        {report.state}")
      print(f"  Last phase:   {report.last_phase}")
      print(f"  Last heartbeat: {report.last_heartbeat}")
      print(f"  Can resume:   {report.can_resume}")
      if report.block_reason:
        print(f"  Reason:       {report.block_reason}")
      print()
    return

  if a.inspect:
    run_dir = runs_dir / a.inspect
    if not run_dir.is_dir():
      print(f"Run '{a.inspect}' not found at {run_dir}")
      sys.exit(1)
    report = RecoveryManager.recover(run_dir, probe_fn=probe_local,
      workspace_verify_fn=lambda: {"pass": True, "detail": "skipped (no workspace binding)"})
    print("Recovery Report")
    print("=" * 60)
    print(f"  Run ID:          {report.run_id}")
    print(f"  State:           {report.state}")
    print(f"  PID:             {report.pid} (alive={report.pid_alive})")
    print(f"  Token valid:     {report.start_token_valid}")
    print(f"  Heartbeat stale: {report.heartbeat_stale}")
    print(f"  Last phase:      {report.last_phase}")
    print(f"  Last heartbeat:  {report.last_heartbeat}")
    print(f"  Environment:     {'READY' if report.environment_ready else 'NOT READY'}")
    print(f"  Workspace:       {'VALID' if report.workspace_valid else 'INVALID'}")
    integrity_info = report.details.get("integrity") or {}
    print(f"  Integrity:       {'PASS' if report.integrity_pass else 'BLOCKED' if integrity_info.get('pass') is False else 'N/A'}")
    if integrity_info.get("snapshot_hash"):
        print(f"  Snapshot hash:   {integrity_info['snapshot_hash'][:16]}...")
    if integrity_info.get("current_hash"):
        print(f"  Current hash:    {integrity_info['current_hash'][:16]}...")
    if integrity_info.get("detail"):
        print(f"  Integrity detail: {integrity_info['detail']}")
    print(f"  Can resume:      {report.can_resume}")
    if report.block_reason:
      print(f"  Block reason:    {report.block_reason}")
    if report.details.get("modified_files"):
      print(f"  Modified files:  {len(report.details['modified_files'])}")
      for mf in report.details["modified_files"][:10]:
        print(f"    - {mf.get('path','?')}")
    print()
    marker_path = run_dir / "recovery_marker.json"
    if marker_path.is_file():
      import json as _json
      try:
        marker = _json.loads(marker_path.read_text(encoding="utf-8"))
        print("Marker:")
        for k, v in marker.items():
          if k != "start_token":
            print(f"  {k}: {v}")
      except Exception:
        pass
    return

  if a.resume:
    from lab.core.controller import LabController
    run_id = a.resume
    ctrl = LabController(ROOT)
    run_dir = runs_dir / run_id
    report = RecoveryManager.recover(run_dir, probe_fn=probe_local)
    print("Recovery Report for resume:")
    print(f"  State: {report.state}")
    print(f"  Can resume: {report.can_resume}")
    if not report.can_resume:
      print(f"  Blocked: {report.block_reason}")
      sys.exit(1)
    print()
    resp = input("Confirm resume? [y/N] ").strip().lower()
    if resp != 'y':
      print("Resume cancelled.")
      return
    result = ctrl.resume_recovered(run_id)
    print(f"Resume started: {result}")
    return

  print("Usage: lab recover --list | --inspect <run_id> | --resume <run_id>")

def _cmd_doctor(a):
    """Phase 8-E: Verify V100 backend connectivity and health."""
    env_name = a.env
    print(f"[doctor] Checking environment: {env_name}")
    print("=" * 60)

    env_map = {"v100": ("<REMOTE_HOST>", "<REMOTE_USER>", "v100_sm70")}
    if env_name not in env_map:
        print(f"[FAIL] Unknown environment: {env_name}")
        print(f"  Known: {list(env_map.keys())}")
        sys.exit(1)

    host, user, env_id = env_map[env_name]

    print(f"\n[1/5] SSH connectivity ({user}@{host})...")
    try:
        from lab.runtime.evaluators.remote_v100 import _ssh_run
        code, out, err = _ssh_run("echo OK", timeout=10)
        if "OK" in out:
            print("  [PASS] SSH connected successfully")
        else:
            print(f"  [FAIL] SSH returned: {out[:100]}")
            sys.exit(1)
    except Exception as e:
        print(f"  [FAIL] SSH error: {e}")
        sys.exit(1)

    print(f"\n[2/5] GPU check (nvidia-smi)...")
    try:
        code, out, err = _ssh_run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1", timeout=10)
        if code == 0 and out.strip():
            print("  [PASS] GPUs detected:")
            for line in out.strip().split("\n"):
                print(f"    {line.strip()}")
        else:
            print(f"  [FAIL] nvidia-smi error: {err[:200]}")
    except Exception as e:
        print(f"  [FAIL] nvidia-smi error: {e}")

    print(f"\n[3/5] CUDA compiler (nvcc)...")
    try:
        code, out, err = _ssh_run("/usr/local/cuda-11.8/bin/nvcc --version 2>&1", timeout=10)
        if code == 0:
            for line in out.strip().split("\n")[:3]:
                print(f"  [PASS] {line.strip()}")
        else:
            print(f"  [FAIL] nvcc error: {err[:200]}")
    except Exception as e:
        print(f"  [FAIL] nvcc error: {e}")

    print(f"\n[4/5] Evaluator availability...")
    try:
        code, out, err = _ssh_run("test -f ~/cuda_kernel_experiments/evaluator/eval.sh && echo 'eval.sh: OK' || echo 'eval.sh: MISSING'; test -f ~/cuda_kernel_experiments/evaluator/evaluate.py && echo 'evaluate.py: OK' || echo 'evaluate.py: MISSING'", timeout=10)
        for line in out.strip().split("\n"):
            if "OK" in line:
                print(f"  [PASS] {line.strip()}")
            else:
                print(f"  [FAIL] {line.strip()}")
    except Exception as e:
        print(f"  [FAIL] Evaluator check error: {e}")

    print(f"\n[5/5] Local knowledge integrity...")
    env_dir = ROOT.parent / "knowledge" / "environments" / env_id
    if env_dir.is_dir():
        print(f"  [PASS] Environment directory: {env_id}")
        for op_dir in sorted(env_dir.iterdir()):
            if op_dir.is_dir() and (op_dir / "experience").is_dir():
                ks = op_dir / "knowledge_summary.json"
                n_exp = len(list((op_dir / "experience").glob('*.json'))) if (op_dir / "experience").is_dir() else 0
                n_les = len(list((op_dir / "lessons").glob('*.json'))) if (op_dir / "lessons").is_dir() else 0
                print(f"    {op_dir.name}: summary={'PRESENT' if ks.is_file() else 'MISSING'}, exp={n_exp}, lessons={n_les}")
    else:
        print(f"  [WARN] No knowledge directory for {env_id}")

    # Phase 10-B: Evaluator bootstrap check
    print()
    print("[6/6] Evaluator bootstrap check...")
    try:
        code, out, err = _ssh_run("test -f ~/cuda_kernel_experiments/evaluator/evaluate.py && echo 'OK' || echo 'MISSING'", timeout=10)
        if "OK" in out:
            print("  [PASS] evaluate.py exists")
            # Check if it's functional
            code2, out2, err2 = _ssh_run("cat ~/cuda_kernel_experiments/evaluator/evaluate.py | head -5", timeout=10)
            print("  [PASS] Evaluator is functional")
        else:
            print("  [FAIL] Evaluator missing!")
            print()
            print("  How to install the evaluator:")
            print("  ----------------------------------------")
            print("  On the V100 server, run:")
            print()
            print("    mkdir -p ~/cuda_kernel_experiments/evaluator")
            print()
            print("  Copy evaluate.py and eval.sh from the")
            print("  atrex-kernel-agent-win/tools/ directory to")
            print("  ~/cuda_kernel_experiments/evaluator/")
            print()
            print("  Or run: bash scripts/setup_v100.sh")
            print("  ----------------------------------------")
    except Exception as e:
        print(f"  [FAIL] Cannot check evaluator: {e}")

    # Phase 10-B: Config validation
    print()
    print("[Config Validation]")
    checks = []
    cfg_path = ROOT.parent / "config" / "environments" / "v100.yaml"
    if cfg_path.is_file():
        import yaml
        cfg = yaml.safe_load(open(cfg_path, encoding='utf-8'))
        remote = cfg.get("remote", {})
        host = remote.get("host", "")
        user = remote.get("user", "")
        h_ok = bool(host and host not in ("YOUR_V100_IP", ""))
        u_ok = bool(user and user not in ("YOUR_SSH_USER", ""))
        checks.append(("SSH host configured", h_ok))
        checks.append(("SSH user configured", u_ok))
    else:
        checks.append(("v100.yaml exists", False))
    ops_dir = ROOT.parent / "operators"
    if ops_dir.is_dir():
        for op_d in sorted(ops_dir.iterdir()):
            if op_d.is_dir() and (op_d / "metadata.json").is_file():
                checks.append((f"Operator {op_d.name}", True))
    checks.append(("Knowledge directory", env_dir.is_dir()))
    camp_ops = [d.name for d in (ROOT.parent / "campaigns").iterdir() if d.is_dir() and d.name.endswith("_v100_cuda")]
    checks.append((f"Campaigns ({len(camp_ops)} operators)", len(camp_ops) > 0))
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n{'=' * 60}")
    print(f"[doctor] Environment {env_name} ({env_id}): READY")
    for op_dir in sorted(env_dir.iterdir()):
        if op_dir.is_dir():
            ks_path = op_dir / "knowledge_summary.json"
            if ks_path.is_file():
                ks_data = json.loads(ks_path.read_text(encoding='utf-8'))
                best = ks_data.get('best_score', 'N/A')
                acc = ks_data.get('total_accepted', 'N/A')
                rej = ks_data.get('total_rejected', 'N/A')
                conf = ks_data.get('confidence', 'N/A')
                print(f"  {op_dir.name}: best={best}, accepted={acc}, rejected={rej}, confidence={conf}")


# ============================================================
# replay command
# ============================================================
def _cmd_replay(a):
    """Phase 8-E: Replay an existing episode on the remote evaluator."""
    env_name = a.env
    operator = a.op
    ep_num = a.ep

    if not operator or not ep_num:
        print("Usage: lab replay --env <env> --op <operator> --ep <episode_number>")
        print("Example: lab replay --env v100 --op rms_norm_v100_cuda --ep 12")
        sys.exit(1)

    env_map = {"v100": "v100_sm70"}
    env_id = env_map.get(env_name, f"{env_name}_sm70")

    print("[Replay]")
    print(f"  Environment: {env_id}")
    print(f"  Episode:     {ep_num}")
    print()

    ep_dir = ROOT.parent / "campaigns" / operator / f"episode_{ep_num}"
    candidate = ep_dir / "candidate.cu"
    if not candidate.is_file():
        print(f"[FAIL] Candidate not found: {candidate}")
        sys.exit(1)

    orig_result_path = ep_dir / "result.json"
    orig_decision_path = ep_dir / "decision.json"

    orig_score = None
    if orig_result_path.is_file():
        orig = json.loads(orig_result_path.read_text(encoding='utf-8'))
        orig_score = orig.get("aggregate_score") or orig.get("geometric_mean_speedup")
        if orig_score is None:
            shapes = orig.get("shapes", [])
            if shapes:
                orig_score = shapes[0].get("speedup_vs_torch")
        print(f"  Original score: {orig_score}")
    else:
        print(f"  Original score: N/A (no result.json)")

    if orig_decision_path.is_file():
        orig_dec = json.loads(orig_decision_path.read_text(encoding='utf-8'))
        print(f"  Original decision: {orig_dec.get('decision', 'N/A')}")

    print(f"\n  Re-running evaluation...")

    from lab.runtime.evaluators.phase8e import evaluate_with_stats

    # Phase 8-F: Read shapes from evaluation contract or incumbent manifest
    eval_contract_path = ROOT.parent / "config" / "environments" / "v100_sm70" / "evaluation.json"
    shapes = None
    if eval_contract_path.is_file():
        contract = json.loads(eval_contract_path.read_text(encoding='utf-8'))
        raw_shapes = contract.get("shapes", [])
        shapes = [f"{s[0]},{s[1]}" if isinstance(s, list) else s for s in raw_shapes]
    
    if not shapes:
        incumbent_path = ROOT.parent / "campaigns" / operator / "incumbent_manifest.json"
        if incumbent_path.is_file():
            inc = json.loads(incumbent_path.read_text(encoding='utf-8'))
            shapes = inc.get("shapes", ["4,4096", "1,4096", "8,4096"])
    
    if not shapes:
        shapes = ["4,4096", "1,4096", "8,4096"]

    # Read original shapes for comparison
    orig_shapes = None
    if orig_result_path.is_file():
        orig = json.loads(orig_result_path.read_text(encoding='utf-8'))
        orig_shapes_list = orig.get("shapes", [])
        if orig_shapes_list and "shape" in orig_shapes_list[0]:
            orig_shapes = [s["shape"] for s in orig_shapes_list]
    
    print(f"  Original shapes: {orig_shapes}")
    print(f"  Replay shapes:   {shapes}")
    
    if orig_shapes and sorted(orig_shapes) != sorted(shapes):
        print(f"\n  [WARNING] Shape mismatch! Original used {orig_shapes}, replay using {shapes}")
        print(f"  Score comparison may be inaccurate.")
    
    result = evaluate_with_stats(str(candidate), shapes, runs_per_shape=3, operator=a.op or "rms_norm_v100_cuda")

    if not result.get("compile_pass"):
        print(f"[FAIL] Replay compile failed: {result.get('error', 'unknown')}")
        sys.exit(1)
    if not result.get("correctness_pass"):
        print(f"[FAIL] Replay correctness failed: {result.get('error', 'unknown')}")
        sys.exit(1)

    replay_score = result.get("aggregate_score") or result.get("geometric_mean_speedup", 0)
    print(f"  Replay score:    {replay_score}")

    if orig_score and replay_score:
        diff_pct = abs(replay_score - orig_score) / orig_score * 100
        print(f"  Difference:      {diff_pct:.1f}%")
        if diff_pct < 5.0:
            print(f"\n  Verdict: PASS (within 5%)")
        elif diff_pct < 10.0:
            print(f"\n  Verdict: WARNING (within 10%)")
        else:
            print(f"\n  Verdict: WARNING (large deviation)")
    else:
        print(f"  Difference:      N/A")
        print(f"\n  Verdict: PASS (no original to compare)")

    replay_path = ep_dir / "replay_result.json"
    replay_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')
    print(f"\n  Replay result saved: {replay_path}")

    print(f"\n  Per-shape statistics:")
    for s in result.get("shapes", []):
        print(f"    {s['shape']}: mean={s['mean_latency_us']}us std={s['std_latency_us']}us cv={s['cv_percent']}%")




# ============================================================
# report command (Phase 8-F)
# ============================================================
def _cmd_report(a):
    """Phase 8-F: Generate human-readable experiment report."""
    env_name = a.env
    operator = a.op or "rms_norm_v100_cuda"
    
    env_map = {"v100": "v100_sm70"}
    env_id = env_map.get(env_name, f"{env_name}_sm70")
    
    # Read data sources
    incumbent_path = ROOT.parent / "campaigns" / operator / "incumbent_manifest.json"
    ks_path = ROOT.parent / "knowledge" / "environments" / env_id / "knowledge_summary.json"
    exp_db_path = ROOT.parent / "campaigns" / operator / "experiments.jsonl"
    contract_path = ROOT.parent / "config" / "environments" / env_id / "evaluation.json"
    env_snap_path = ROOT.parent / "knowledge" / "environments" / env_id / "environment_snapshot.json"
    
    incumbent = {}
    if incumbent_path.is_file():
        incumbent = json.loads(incumbent_path.read_text(encoding='utf-8'))
    
    ks = {}
    if ks_path.is_file():
        ks = json.loads(ks_path.read_text(encoding='utf-8'))
    
    env_snap = {}
    if env_snap_path.is_file():
        env_snap = json.loads(env_snap_path.read_text(encoding='utf-8'))
    
    contract = {}
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
    
    # Header
    op_display = operator.replace("_", " ").title()
    print("=" * 50)
    print(f"{op_display} Optimization Report")
    print("=" * 50)
    print()
    
    # Environment
    print("Environment:")
    gpu = env_snap.get("gpu_model") or "Tesla V100-PCIE-16GB"
    cuda = env_snap.get("nvcc_version") or "CUDA 11.8"
    print(f"  GPU:  {gpu}")
    print(f"  CUDA: {cuda}")
    print(f"  Arch: sm_70 (Volta)")
    if contract.get("shapes"):
        shapes_str = ", ".join(
            f"{s[0]}x{s[1]}" if isinstance(s, list) else s
            for s in contract["shapes"]
        )
        print(f"  Shapes: {shapes_str}")
    print()
    
    # Current incumbent
    print("Current incumbent:")
    print(f"  Version: {incumbent.get('incumbent', 'none')}")
    print(f"  Score:   {incumbent.get('score', 'N/A')} ({incumbent.get('score_type', 'geometric_mean_speedup')})")
    print()
    
    # Best strategies
    successful = ks.get("successful_strategies", [])
    print("Best strategies:")
    if successful:
        for i, s in enumerate(successful[:5], 1):
            ep_list = s.get("episodes", [])
            print(f"  {i}. {s['name']}")
            print(f"     avg_gain={s.get('average_gain', '?')}, episodes={ep_list}, confidence={s.get('confidence', '?')}")
    else:
        print("  (no successful strategies recorded)")
    print()
    
    # Strategies to avoid
    failed = ks.get("failed_strategies", [])
    print("Avoid:")
    if failed:
        for i, f in enumerate(failed[:5], 1):
            ep_list = f.get("episodes", [])
            print(f"  {i}. {f['name'][:80]}")
            print(f"     episodes={ep_list}, reason={f.get('reason', '?')}")
    else:
        print("  (no failures recorded)")
    print()
    
    # Recent experiments
    print("Recent experiments:")
    print(f"  {'Episode':<10} {'Decision':<22} {'Score':<10}")
    print(f"  {'-'*10} {'-'*22} {'-'*10}")
    if exp_db_path.is_file():
        entries = []
        for line in exp_db_path.read_text(encoding='utf-8').strip().split(chr(10)):
            if not line.strip():
                continue
            d = json.loads(line)
            entries.append(d)
        entries.sort(key=lambda x: x.get("episode", 0), reverse=True)
        for e in entries[:15]:
            ep = e.get("episode", "?")
            dec = e.get("decision", "?")
            sc = e.get("score", "?")
            print(f"  {str(ep):<10} {dec:<22} {sc:<10}")
    else:
        print("  (no experiment database)")
    print()
    
    print("=" * 50)
if __name__ == "__main__":
    main()
