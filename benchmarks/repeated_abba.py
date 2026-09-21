"""Five-batch robustness check for the frozen V2b incumbent and V3 candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch


SHAPES = ((256, 4096), (1024, 4096), (4096, 4096))


def load_model(path: Path, tag: str):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module.Model(1, 4096).cuda()


def timed(fn, gate, up, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn(gate, up)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn(gate, up)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) * 1000.0 / repeats)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    incumbent = load_model(args.incumbent, "aka_robust_incumbent")
    candidate = load_model(args.candidate, "aka_robust_candidate")
    per_shape = {}
    for m, d in SHAPES:
        rows = []
        for batch in range(args.batches):
            gate = torch.randn((m, d), device="cuda", dtype=torch.float16)
            up = torch.randn_like(gate)
            expected = torch.nn.functional.silu(gate) * up
            actual = candidate(gate, up)
            diff = (actual - expected).abs()
            values = [
                timed(incumbent, gate, up, args.warmup, args.repeats),
                timed(candidate, gate, up, args.warmup, args.repeats),
                timed(candidate, gate, up, args.warmup, args.repeats),
                timed(incumbent, gate, up, args.warmup, args.repeats),
            ]
            incumbent_us = (values[0] + values[3]) / 2.0
            candidate_us = (values[1] + values[2]) / 2.0
            rows.append({
                "batch": batch + 1,
                "A1_us": values[0], "B1_us": values[1],
                "B2_us": values[2], "A2_us": values[3],
                "incumbent_mean_us": incumbent_us,
                "candidate_mean_us": candidate_us,
                "speedup": incumbent_us / candidate_us,
                "max_abs": float(diff.max().item()),
                "max_rel": float((diff / expected.abs().clamp_min(1e-6)).max().item()),
            })
        speeds = [row["speedup"] for row in rows]
        per_shape[f"m{m}_d{d}"] = {
            "batches": rows,
            "mean_speedup": statistics.mean(speeds),
            "median_speedup": statistics.median(speeds),
            "std_speedup": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
            "candidate_wins": sum(speed > 1.0 for speed in speeds),
            "wins_total": len(speeds),
            "incumbent_total_us": sum(row["incumbent_mean_us"] for row in rows),
            "candidate_total_us": sum(row["candidate_mean_us"] for row in rows),
        }

    means = [row["mean_speedup"] for row in per_shape.values()]
    total_inc = sum(row["incumbent_total_us"] for row in per_shape.values())
    total_cand = sum(row["candidate_total_us"] for row in per_shape.values())
    payload = {
        "schema_version": 1,
        "protocol": "five_independent_same_process_ABBA_batches",
        "order": "A1,B1,B2,A2",
        "batches": args.batches,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "shapes": per_shape,
        "aggregate": {
            "arithmetic_mean_speedup": statistics.mean(means),
            "geometric_mean_speedup": math.prod(means) ** (1.0 / len(means)),
            "total_time_ratio": total_inc / total_cand,
        },
        "promotion": {
            "policy_decision": "ACCEPT" if statistics.mean(means) > 1.0 else "REJECT",
            "engineering_decision": "UNDECIDED",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
