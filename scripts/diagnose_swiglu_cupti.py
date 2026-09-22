"""Read-only CUPTI capability inventory and minimal profiler probe."""
from __future__ import annotations

import ctypes.util
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    record = {
        "interpreter": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "torch_profiler_invocation": "torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True, profile_memory=True)",
        "cupti_find_library": {name: ctypes.util.find_library(name) for name in ("cupti", "cupti64_2026.1", "cupti64_12")},
    }
    try:
        record["device_properties"] = str(torch.cuda.get_device_properties(0))
        record["driver_version"] = torch._C._cuda_getDriverVersion()
    except Exception as exc:
        record["device_query_error"] = f"{type(exc).__name__}: {exc}"
    try:
        runtime = torch.cuda.cudart()
        version = runtime.cudaRuntimeGetVersion()
        record["cuda_runtime_get_version"] = version[1] if isinstance(version, tuple) else version
    except Exception as exc:
        record["runtime_query_error"] = f"{type(exc).__name__}: {exc}"
    for command in (["where.exe", "cupti64*.dll"], ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]):
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
            record["command:" + " ".join(command)] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as exc:
            record["command:" + " ".join(command)] = {"error": f"{type(exc).__name__}: {exc}"}
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
