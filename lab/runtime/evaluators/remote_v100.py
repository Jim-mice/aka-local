"""Remote V100 CUDA Kernel Evaluator (Phase 8-B).

Evaluates standalone .cu files on a remote V100 server via SSH.
Implements the standard Evaluator Protocol for aka-local integration.

Phase 8-B additions:
- Multi-shape evaluation with geometric mean scoring
- NSYS profile download and lightweight parsing
- Backward-compatible single-shape API preserved
"""

import json
import os
import re
import statistics
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import paramiko
from scp import SCPClient

# -- config --
# Read from config; fallback for backward compat
def _load_ssh_config():
    cfg_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "environments" / "v100.yaml"
    if cfg_path.is_file():
        import yaml
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            remote = cfg.get("remote", {})
            return remote.get("host", "<REMOTE_HOST>"), remote.get("user", "<REMOTE_USER>")
        except Exception:
            pass
    return "<REMOTE_HOST>", "<REMOTE_USER>"

V100_HOST, V100_USER = _load_ssh_config()
V100_EVAL_DIR = "~/cuda_kernel_experiments/evaluator"
V100_EVAL_SH = f"{V100_EVAL_DIR}/eval.sh"
V100_WORK_DIR = "~/aka_remote_jobs"

# Multi-shape defaults for RMSNorm
DEFAULT_SHAPES = ["4,4096", "1,4096", "8,4096"]


def _get_password() -> str:
    pw = os.environ.get("AKA_V100_PASSWORD", "")
    if not pw:
        raise RuntimeError("AKA_V100_PASSWORD not set")
    return pw


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- SSH helpers --

def _ssh_client() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(V100_HOST, username=V100_USER, password=_get_password(), timeout=15)
    return c


def _scp_upload(local_path: str, remote_path: str) -> None:
    c = _ssh_client()
    try:
        with SCPClient(c.get_transport()) as scp:
            scp.put(local_path, remote_path)
    finally:
        c.close()


def _ssh_run(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    c = _ssh_client()
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return exit_code, out, err
    finally:
        c.close()


def _scp_download(remote_path: str, local_path: str) -> bool:
    """Download a single file from V100. Returns True on success."""
    c = _ssh_client()
    try:
        with SCPClient(c.get_transport()) as scp:
            scp.get(remote_path, local_path)
        return True
    except Exception:
        return False
    finally:
        c.close()


# -- Evaluator Protocol --

class RemoteV100Evaluator:
    """Evaluates a standalone .cu candidate on a remote V100 server."""

    def __init__(self, candidate_path: str, shape: str = "4,4096", operator: str = "rms_norm_v100_cuda"):
        self.candidate_path = Path(candidate_path)
        self.shape = shape
        self.operator = operator
        self.job_id = uuid.uuid4().hex[:12]
        self.remote_job_dir = f"{V100_WORK_DIR}/{self.job_id}"
        self._result: Optional[dict] = None
        self._compile_ok: bool = False
        self._error: str = ""

    def prepare(self) -> bool:
        try:
            _ssh_run(f"mkdir -p {self.remote_job_dir}", timeout=10)
            remote_cu = f"{self.remote_job_dir}/candidate.cu"
            _scp_upload(str(self.candidate_path), remote_cu)
            return True
        except Exception as e:
            self._error = f"prepare failed: {e}"
            return False

    def compile(self) -> bool:
        try:
            remote_cu = f"{self.remote_job_dir}/candidate.cu"
            out_json = f"{self.remote_job_dir}/result.json"

            cmd = (
                f"cd {V100_EVAL_DIR} && "
                f"bash eval.sh {remote_cu} --shape {self.shape} --op {self.operator} "
                f"--no-profile -o {out_json} 2>&1"
            )
            exit_code, stdout, stderr = _ssh_run(cmd, timeout=60)

            local_tmp = Path(tempfile.gettempdir()) / f"v100_result_{self.job_id}.json"
            if not _scp_download(out_json, str(local_tmp)):
                exit_code2, raw, _ = _ssh_run(f"cat {out_json} 2>/dev/null || echo '{{}}'", timeout=10)
                if raw.strip() and raw.strip() != "{}":
                    local_tmp.write_text(raw.strip(), encoding="utf-8")
                else:
                    self._error = f"no result.json from V100; stdout={stdout[:500]}"
                    return False

            if local_tmp.is_file():
                self._result = json.loads(local_tmp.read_text(encoding="utf-8"))
                self._compile_ok = self._result.get("compile", False)
                if not self._compile_ok:
                    self._error = self._result.get("compile_error", "unknown compile error")
                return self._compile_ok
            else:
                self._error = f"result.json not generated"
                return False

        except Exception as e:
            self._error = f"compile exception: {e}"
            return False

    def check_correctness(self) -> bool:
        if not self._result:
            return False
        ok = self._result.get("correctness", False)
        if not ok:
            self._error = f"correctness failed; max_error={self._result.get('max_error', '?')}"
        return ok

    def benchmark(self) -> Optional[dict]:
        if not self._result:
            return None
        return {
            "latency_us": self._result.get("latency_us"),
            "torch_naive_latency_us": self._result.get("torch_naive_latency_us"),
            "torch_optimized_latency_us": self._result.get("torch_optimized_latency_us"),
            "speedup_vs_naive": self._result.get("speedup_vs_naive"),
            "speedup_vs_torch": self._result.get("speedup_vs_torch"),
            "source": "remote-v100-ssh",
            "gpu": "Tesla V100-PCIE-16GB",
            "arch": "sm_70",
            "timestamp": utcnow(),
        }

    def profile(self) -> Optional[dict]:
        if not self._result:
            return None
        nsys = self._result.get("nsys_report", "")
        return {"nsys_report": nsys} if nsys else None

    def static_evidence(self) -> dict:
        if not self._result:
            return {"compile": False, "error": self._error}
        return {
            "compile": self._result.get("compile", False),
            "compile_error": self._result.get("compile_error", ""),
            "nvcc_cmd": self._result.get("nvcc_cmd", ""),
            "correctness": self._result.get("correctness", False),
            "max_error": self._result.get("max_error"),
        }

    def cleanup(self) -> None:
        try:
            _ssh_run(f"rm -rf {self.remote_job_dir}", timeout=10)
        except Exception:
            pass

    def get_result(self) -> Optional[dict]:
        return self._result

    def get_error(self) -> str:
        return self._error


# -- Single-shape convenience (backward compatible) --

def evaluate_candidate(candidate_path: str, shape: str = "4,4096", operator: str = "rms_norm_v100_cuda") -> dict:
    """Evaluate a candidate.cu on V100 for one shape.
    Returns backward-compatible Evidence dict."""
    ev = RemoteV100Evaluator(candidate_path, shape, operator)
    result = {
        "compile_pass": False,
        "correctness_pass": False,
        "speedup": None,
        "latency_us": None,
        "evidence": {},
        "error": "",
    }

    if not ev.prepare():
        result["error"] = ev.get_error()
        return result

    if not ev.compile():
        result["error"] = ev.get_error()
        result["evidence"] = ev.static_evidence()
        return result

    result["compile_pass"] = True

    if ev.check_correctness():
        result["correctness_pass"] = True
    else:
        result["error"] = ev.get_error()
        result["evidence"] = ev.static_evidence()
        return result

    bench = ev.benchmark()
    if bench:
        result["speedup"] = bench.get("speedup_vs_torch")
        result["latency_us"] = bench.get("latency_us")
        result["evidence"] = bench
        result["evidence"].update(ev.static_evidence())

    ev.cleanup()
    return result


# -- Phase 8-B: Multi-shape evaluation --

def evaluate_candidate_multi_shape(
    candidate_path: str,
    shapes: list = None,
    operator: str = "rms_norm_v100_cuda",
    with_profile: bool = False
) -> dict:
    """Evaluate candidate.cu across multiple shapes on V100.

    Returns:
        {
            "compile_pass": bool,
            "correctness_pass": bool,
            "shapes": [{"shape": "B,H", "latency_us": ..., "speedup": ...}],
            "aggregate_score": float,
            "geometric_mean_speedup": float,
            "speedup": float,          # backward-compat: default-shape speedup
            "latency_us": float,        # backward-compat
            "evidence": dict,           # backward-compat
            "error": str,
            "profile_available": bool,
            "profile_summary": dict or None,
        }
    """
    if shapes is None:
        shapes = DEFAULT_SHAPES

    result = {
        "compile_pass": False,
        "correctness_pass": False,
        "shapes": [],
        "aggregate_score": 0.0,
        "geometric_mean_speedup": 1.0,
        "speedup": None,
        "latency_us": None,
        "evidence": {},
        "error": "",
        "profile_available": False,
        "profile_summary": None,
    }

    candidate_path = str(candidate_path)

    # Phase 1: Compile once (use first shape for compile check)
    primary_shape = shapes[0]
    ev = RemoteV100Evaluator(candidate_path, primary_shape, operator=operator)
    if not ev.prepare():
        result["error"] = ev.get_error()
        return result

    if not ev.compile():
        result["error"] = ev.get_error()
        result["evidence"] = ev.static_evidence()
        return result

    result["compile_pass"] = True

    # Phase 2: Correctness on primary shape
    if not ev.check_correctness():
        result["error"] = ev.get_error()
        result["evidence"] = ev.static_evidence()
        result["shapes"] = [{"shape": primary_shape, "correctness": False}]
        return result

    result["correctness_pass"] = True

    # Phase 3: Benchmark across all shapes
    speedups = []
    all_latencies = {}
    all_correct = True  # Phase 9.5: track correctness across ALL shapes

    for shape in shapes:
        sev = RemoteV100Evaluator(candidate_path, shape, operator)
        if not sev.prepare():
            result["shapes"].append({"shape": shape, "error": "prepare failed"})
            continue
        if not sev.compile():
            result["shapes"].append({"shape": shape, "error": sev.get_error()})
            continue
        if not sev.check_correctness():
            all_correct = False
            result["shapes"].append({"shape": shape, "correctness": False, "error": sev.get_error()})
            continue

        bench = sev.benchmark()
        shape_result = {
            "shape": shape,
            "latency_us": bench.get("latency_us") if bench else None,
            "speedup_vs_torch": bench.get("speedup_vs_torch") if bench else None,
            "speedup_vs_naive": bench.get("speedup_vs_naive") if bench else None,
            "correctness": True,
        }
        result["shapes"].append(shape_result)

        if bench and bench.get("speedup_vs_torch"):
            speedups.append(float(bench["speedup_vs_torch"]))
        if bench and bench.get("latency_us"):
            all_latencies[shape] = bench["latency_us"]

        sev.cleanup()

    # Phase 4: Compute aggregate score AND validate all-shape correctness
    # Phase 9.5: If any shape failed correctness, mark overall as failed
    if not all_correct:
        result["correctness_pass"] = False
        result["error"] = "correctness failed on one or more shapes"
    
    if speedups:
        result["geometric_mean_speedup"] = round(
            statistics.geometric_mean(speedups), 3
        )
        result["aggregate_score"] = result["geometric_mean_speedup"]

    # Backward-compatible fields from primary shape
    primary = result["shapes"][0] if result["shapes"] else {}
    result["speedup"] = primary.get("speedup_vs_torch")
    result["latency_us"] = primary.get("latency_us")

    # Compile evidence from primary
    result["evidence"] = ev.static_evidence()
    bench = ev.benchmark()
    if bench:
        result["evidence"].update(bench)

    ev.cleanup()

    # Phase 5: NSYS profile (if requested)
    if with_profile:
        profile_ev = RemoteV100Evaluator(candidate_path, primary_shape)
        if profile_ev.prepare():
            # Run without --no-profile to trigger NSYS
            remote_cu = f"{profile_ev.remote_job_dir}/candidate.cu"
            out_json = f"{profile_ev.remote_job_dir}/result.json"
            cmd = (
                f"cd {V100_EVAL_DIR} && "
                f"bash eval.sh {remote_cu} --shape {primary_shape} "
                f"-o {out_json} 2>&1"
            )
            _ssh_run(cmd, timeout=120)

            # Try to download NSYS report
            local_nsys = Path(tempfile.gettempdir()) / f"v100_nsys_{uuid.uuid4().hex[:8]}.nsys-rep"
            bn = os.path.splitext(os.path.basename(str(candidate_path)))[0]
            remote_nsys = f"{profile_ev.remote_job_dir}/nsys_{bn}.nsys-rep"

            if _scp_download(remote_nsys, str(local_nsys)):
                summary = _parse_nsys_summary(local_nsys)
                result["profile_available"] = True
                result["profile_summary"] = summary
            else:
                result["profile_summary"] = {"available": False, "reason": "nsys download failed"}

            profile_ev.cleanup()

    return result


def _parse_nsys_summary(nsys_path: str) -> dict:
    """Lightweight NSYS summary extraction.
    Does NOT implement full NCU parsing — just what we can get from nsys stats.
    """
    summary = {
        "source": "nsys",
        "kernel_time_us": None,
        "launch_overhead_us": None,
        "memory_behavior": "unknown",
    }
    try:
        # Try nsys stats CLI (local would need nsys; skip gracefully)
        import subprocess
        r = subprocess.run(
            ["nsys", "stats", "--report", "cuda_gpu_kern_sum", nsys_path],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0 and r.stdout:
            # Extract kernel time from nsys stats output
            for line in r.stdout.split("\n"):
                if "Total" in line and "Time" in line:
                    m = re.search(r"([\d.]+)\s*(us|ms|s)", line)
                    if m:
                        val = float(m.group(1))
                        unit = m.group(2)
                        if unit == "ms":
                            val *= 1000
                        elif unit == "s":
                            val *= 1_000_000
                        summary["kernel_time_us"] = round(val, 2)
                        break
            # Memory behavior heuristic
            if "Device Memory" in r.stdout:
                summary["memory_behavior"] = "device_memory_bound"
            elif "Shared Memory" in r.stdout:
                summary["memory_behavior"] = "shared_memory_heavy"
    except Exception:
        pass

    # Fallback: try nsys stats on the remote
    try:
        code, out, err = _ssh_run(
            f"nsys stats --report cuda_gpu_kern_sum {nsys_path} 2>/dev/null || echo 'NSYS_UNAVAILABLE'",
            timeout=30
        )
        if "NSYS_UNAVAILABLE" not in out and out.strip():
            for line in out.split("\n"):
                if "Total" in line and "Time" in line:
                    m = re.search(r"([\d.]+)\s*(us|ms|s)", line)
                    if m:
                        val = float(m.group(1))
                        unit = m.group(2)
                        if unit == "ms":
                            val *= 1000
                        elif unit == "s":
                            val *= 1_000_000
                        summary["kernel_time_us"] = round(val, 2)
                        break
    except Exception:
        pass

    return summary


# -- CLI entry --

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Remote V100 CUDA Evaluator (Phase 8-B)")
    ap.add_argument("candidate", help="Path to candidate.cu")
    ap.add_argument("--shape", default="4,4096", help="Single shape: BATCH,HIDDEN")
    ap.add_argument("--shapes", nargs="*", default=None, help="Multiple shapes: '4,4096 1,4096 8,4096'")
    ap.add_argument("--multi", action="store_true", help="Use multi-shape evaluation")
    ap.add_argument("--profile", action="store_true", help="Request NSYS profile")
    args = ap.parse_args()

    print(f"[V100] Evaluating: {args.candidate}")
    if args.multi or args.shapes:
        shapes = args.shapes if args.shapes else DEFAULT_SHAPES
        res = evaluate_candidate_multi_shape(args.candidate, shapes, with_profile=args.profile)
    else:
        res = evaluate_candidate(args.candidate, args.shape)
    print(json.dumps(res, indent=2, ensure_ascii=False))
