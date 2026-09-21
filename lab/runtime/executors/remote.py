"""Remote V100 executor — wraps lab.runtime.evaluators.remote_v100.

Provides both standalone CLI evaluation and campaign-integrated execution.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


class SSHExecutor:
    """Base SSH executor interface."""

    def __init__(self, credential_store=None):
        self.credential_store = credential_store

    def probe(self, *args, **kwargs):
        from lab.runtime.evaluators.remote_v100 import _ssh_run
        try:
            code, out, err = _ssh_run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", timeout=15)
            if code == 0:
                return {"status": "OK", "gpu_info": out.strip(), "write": False}
            return {"status": "UNREACHABLE", "error": err.strip(), "write": False}
        except Exception as e:
            return {"status": "UNREACHABLE", "error": str(e), "write": False}

    def prepare_workspace(self, *args, **kwargs):
        return {"status": "READY", "write": False}

    def sync_candidate(self, *args, **kwargs):
        return {"status": "READY", "write": False}

    def reconcile(self, *args, **kwargs):
        return {"status": "READY", "write": False}


class RemoteEvaluator(SSHExecutor):
    """Wraps remote_v100 evaluator with executor-compatible interface."""

    def run_correctness(self, candidate_path: str, **kwargs):
        from lab.runtime.evaluators.remote_v100 import RemoteV100Evaluator
        ev = RemoteV100Evaluator(candidate_path)
        if not ev.prepare():
            return {"status": "FAILED", "error": ev.get_error()}
        if not ev.compile():
            return {"status": "COMPILE_FAILED", "error": ev.get_error(), "evidence": ev.static_evidence()}
        ok = ev.check_correctness()
        ev.cleanup()
        return {"status": "PASSED" if ok else "CORRECTNESS_FAILED", "compile": True, "correctness": ok}

    def run_benchmark(self, candidate_path: str, **kwargs):
        from lab.runtime.evaluators.remote_v100 import evaluate_candidate
        result = evaluate_candidate(candidate_path)
        return {"status": "PASSED" if result.get("speedup") else "FAILED", **result}

    def run_profile(self, candidate_path: str, **kwargs):
        from lab.runtime.evaluators.remote_v100 import RemoteV100Evaluator
        ev = RemoteV100Evaluator(candidate_path)
        if not ev.prepare():
            return {"status": "FAILED", "error": ev.get_error()}
        if not ev.compile():
            return {"status": "COMPILE_FAILED", "error": ev.get_error()}
        prof = ev.profile()
        ev.cleanup()
        return {"status": "PASSED" if prof else "NO_PROFILE", "profile": prof}

    def fetch_artifacts(self, job_id: str, **kwargs):
        return {"status": "READY", "job_id": job_id}

    def fetch_events(self, job_id: str, **kwargs):
        return {"status": "READY", "job_id": job_id, "events": []}
