"""Phase 8-D: Reproducibility, Auditability, Operator Scalability."""
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def sha256_hex(data):
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def file_sha256(p):
    if not p.is_file():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

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

# Task 1: Episode reproducibility bundle
def build_episode_manifest(episode_num, operator, candidate_path, contract_shapes, contract=None, evaluation_fingerprint_value=None):
    env_snap_path = ROOT / "knowledge" / "environments" / "v100_sm70" / "environment_snapshot.json"
    env_snap = read_json(env_snap_path, {})
    return {
        "episode": episode_num,
        "operator": operator,
        "environment": {
            "id": "v100_sm70",
            "gpu": env_snap.get("gpu_model", "Tesla V100-PCIE-16GB"),
            "cuda": "11.8",
            "driver": env_snap.get("driver_version", "unknown"),
            "nvcc": env_snap.get("nvcc_version", "unknown"),
        },
        "operator_contract": {
            "name": operator,
            "interface": "standalone_cuda",
            "entry": "launch_kernel",
        },
        # ``contract_hash`` is retained only for legacy readers.  It was a
        # shape-list digest and must never be interpreted as a semantic ABI
        # contract.  New artifacts carry the explicitly named fields below.
        "contract_hash": sha256_hex(json.dumps({"shapes": contract_shapes}, sort_keys=True)),
        "legacy_contract_hash_kind": "evaluation_shapes_only",
        "semantic_contract_sha256": contract.semantic_contract_sha256 if contract else None,
        "evaluation_fingerprint": evaluation_fingerprint_value,
        "interface_entry": contract.entry if contract else None,
        "interface_arguments": contract.arguments if contract else None,
        "result_schema_version": 2,
        "candidate_hash": file_sha256(Path(candidate_path)),
        "baseline_hash": file_sha256(ROOT / "operators" / operator / "reference.cu"),
        "agent": {
            "model": "gpt-5.6-luna",
        },
        "evaluation": {
            "shapes": contract_shapes,
            "score_type": "geometric_mean_speedup",
        },
        "created_time": utcnow(),
    }

# Task 2: Replay command
def replay_episode(operator, episode_num):
    ep_dir = ROOT / "campaigns" / operator / f"episode_{episode_num}"
    candidate_cu = ep_dir / "candidate.cu"
    original_result = ep_dir / "result.json"
    if not candidate_cu.is_file():
        print(f"ERROR: candidate.cu not found in {ep_dir}")
        return 1
    original = read_json(original_result, {})
    original_score = original.get("geometric_mean_speedup") or original.get("speedup", 0)
    print(f"[Replay] Episode {episode_num}")
    print(f"  Original score: {original_score}")
    manifest = read_json(ep_dir / "episode_manifest.json", {})
    shapes = manifest.get("evaluation", {}).get("shapes", manifest.get("shapes", ["4,4096"]))
    from lab.runtime.evaluators.remote_v100 import evaluate_candidate_multi_shape
    new_result = evaluate_candidate_multi_shape(str(candidate_cu), shapes, with_profile=False)
    new_score = new_result.get("geometric_mean_speedup", 0)
    print(f"  New score: {new_score}")
    if original_score and new_score and float(original_score) > 0:
        diff_pct = abs(float(new_score) - float(original_score)) / float(original_score) * 100
        print(f"  Difference: {diff_pct:.2f}%")
        if diff_pct < 5.0:
            print(f"\n  [PASS] Reproducible within 5% tolerance")
        elif diff_pct < 15.0:
            print(f"\n  [WARNING] Environment drift detected ({diff_pct:.1f}%)")
        else:
            print(f"\n  [FAIL] Significant drift ({diff_pct:.1f}%)")
            return 1
    else:
        print(f"\n  [OK] Cannot compare (missing original score)")
    write_json(ep_dir / "replay_result.json", {
        "episode": episode_num, "original_score": original_score,
        "new_score": new_score, "replayed_at": utcnow()
    })
    return 0

# Task 3: Environment snapshot
def collect_environment_snapshot():
    from lab.runtime.evaluators.remote_v100 import _ssh_run
    snap = {}
    try:
        c, out, err = _ssh_run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1", timeout=10)
        if c == 0:
            parts = out.strip().split(", ")
            snap["gpu_model"] = parts[0] if len(parts) > 0 else "unknown"
            snap["driver_version"] = parts[1] if len(parts) > 1 else "unknown"
            snap["memory_mib"] = parts[2] if len(parts) > 2 else "unknown"
        c, out, err = _ssh_run("/usr/local/cuda-11.8/bin/nvcc --version 2>&1", timeout=10)
        for line in out.split("\n"):
            if "release" in line:
                snap["nvcc_version"] = line.strip()
                break
        c, out, err = _ssh_run("uname -r", timeout=10)
        snap["kernel"] = out.strip() if c == 0 else "unknown"
    except Exception as e:
        snap["error"] = str(e)
    snap["collected_at"] = utcnow()
    path = ROOT / "knowledge" / "environments" / "v100_sm70" / "environment_snapshot.json"
    write_json(path, snap)
    return snap

# Task 4: Enhanced knowledge summary with strategies
def _extract_keywords(text):
    keywords = {
        "warp shuffle": "warp shuffle reduction",
        "shared memory": "shared memory optimization",
        "coalesced": "coalesced memory access",
        "fma": "fused multiply-add",
        "rsqrt": "reciprocal square root",
        "reduction": "parallel reduction",
    }
    found = []
    for key, label in keywords.items():
        if key in text.lower():
            found.append(label)
    return found

# Task 5: Operator abstraction
def list_operators():
    ops = []
    ops_dir = ROOT / "operators"
    if ops_dir.is_dir():
        for d in sorted(ops_dir.iterdir()):
            meta_path = d / "metadata.json"
            if meta_path.is_file():
                meta = read_json(meta_path, {})
                meta["_directory"] = str(d.name)
                ops.append(meta)
    return ops

def load_operator_meta(operator_name):
    path = ROOT / "operators" / operator_name / "metadata.json"
    return read_json(path, {"operator": operator_name, "interface": "standalone_cuda"})
