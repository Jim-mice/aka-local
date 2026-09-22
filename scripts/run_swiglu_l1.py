"""Run the D-side SwiGLU L1 import/injection smoke without editing Megatron."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

from lab.runtime.integrations.swiglu_l1 import SwiGLUIntegrationResult, file_sha256, write_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    contract = repo / "targets/megatron_5be9626/integration_contracts/swiglu.json"
    candidate = repo / "lab/runtime/integrations/swiglu_l1.py"
    source_commit = None
    try:
        source_commit = subprocess.check_output(["git", "-C", str(args.megatron_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        source_commit = f"UNKNOWN:{exc.__class__.__name__}"
    environment = {"python": sys.version, "torch": "missing", "distributed_invariants": False}
    try:
        sys.path.insert(0, str(args.megatron_root))
        importlib.import_module("megatron.core.transformer.mlp")
        environment["torch"] = "imported"
        status = "INTEGRATION_BLOCKED"
        reason = "torch import succeeded but executable model/config/input are not supplied"
    except Exception as exc:  # import smoke must report the concrete blocker
        status = "INTEGRATION_BLOCKED"
        reason = f"{type(exc).__name__}: {exc}"
    result = SwiGLUIntegrationResult(
        source_commit=source_commit,
        integration_contract_hash=file_sha256(contract),
        candidate_hash=file_sha256(candidate),
        captured_shape=None,
        forward_correctness={"status": "RUNTIME_BLOCKED"},
        backward_correctness={"status": "BACKWARD_RUNTIME_BLOCKED"},
        replacement_invocations=0,
        baseline_target_invocations=0,
        fallback_detected=None,
        runtime_environment={**environment, "blocker": reason},
        timing_if_available=None,
        status=status,
        evidence={"classification": "HARNESS_IMPLEMENTED", "runtime": "RUNTIME_BLOCKED", "blocker": reason},
    )
    write_result(args.artifact_root / "swiglu_integration_result.json", result)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
