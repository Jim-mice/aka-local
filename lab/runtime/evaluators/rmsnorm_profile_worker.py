"""One official RMSNorm shape workload for Nsight Compute.

This worker owns exactly one Python/CUDA process.  NCU launches this module
directly so kernel filtering, launch skip and launch count apply to the actual
candidate kernel rather than a child process.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


ATREX = Path(r"<LOCAL_USER_HOME>\projects\atrex-bench")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="one official RMSNorm candidate shape for NCU")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--shape-id", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--metadata-output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("warmup must be >= 0 and iterations must be >= 1")
    shapes = json.loads((ATREX / "data" / "rms_norm" / "shapes.json").read_text(encoding="utf-8"))
    key = str(args.shape_id)
    if key not in shapes:
        raise SystemExit(f"unknown official RMSNorm shape id: {args.shape_id}")
    candidate = args.candidate.resolve()
    module = load_module(candidate, "rmsnorm_profile_candidate")
    inputs = load_module(ATREX / "data" / "rms_norm" / "input.py", "rmsnorm_profile_input")
    spec = shapes[key]
    input_kwargs, init_kwargs = spec["input_kwargs"], spec["init_kwargs"]
    model = module.Model(**init_kwargs).cuda().eval()
    values = inputs._make_inputs(**input_kwargs)
    with torch.inference_mode():
        # NCU receives one direct stream of candidate launches.  The adapter
        # filters the known kernel symbol and skips exactly these warmups.
        for _ in range(args.warmup):
            model(**values)
        torch.cuda.synchronize()
        for _ in range(args.iterations):
            model(**values)
        torch.cuda.synchronize()
    metadata = {
        "kind": "RMSNORM_ONE_SHAPE_PROFILE_WORKER",
        "profile_measurement": True,
        "not_latency_benchmark": True,
        "shape_id": args.shape_id,
        "token_count": input_kwargs.get("token_count"),
        "hidden_size": input_kwargs.get("hidden_size"),
        "input_kwargs": input_kwargs,
        "init_kwargs": init_kwargs,
        "candidate": str(candidate),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "gpu": torch.cuda.get_device_name(),
        "architecture": list(torch.cuda.get_device_capability()),
        "expected_kernel_symbol": "rmsnorm_row_kernel",
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
