"""Mechanical same-process ABBA benchmark for the local SwiGLU episode.

This contains no model or agent calls. It loads the two supplied Model
wrappers, allocates one input pair per shape, and measures A/B/B/A with CUDA
events. The result is a JSON object written to the requested path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch


SHAPES = ((256, 4096), (1024, 4096), (4096, 4096))


def load_model_module(path: Path, tag: str) -> Any:
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def timed(fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], gate: torch.Tensor, up: torch.Tensor, warmup: int, repeats: int) -> float:
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
    parser = argparse.ArgumentParser(description="SwiGLU incumbent/candidate CUDA ABBA")
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    incumbent_module = load_model_module(args.incumbent, "aka_episode_incumbent")
    candidate_module = load_model_module(args.candidate, "aka_episode_candidate")
    incumbent = incumbent_module.Model(1, 4096).cuda()
    candidate = candidate_module.Model(1, 4096).cuda()

    rows: dict[str, Any] = {}
    for m, d in SHAPES:
        gate = torch.randn((m, d), device="cuda", dtype=torch.float16)
        up = torch.randn_like(gate)
        expected = torch.nn.functional.silu(gate) * up
        actual = candidate(gate, up)
        diff = (actual - expected).abs()
        max_abs = float(diff.max().item())
        max_rel = float((diff / expected.abs().clamp_min(1e-6)).max().item())

        a = lambda x, y: incumbent(x, y)
        b = lambda x, y: candidate(x, y)
        values = [
            timed(a, gate, up, args.warmup, args.repeats),
            timed(b, gate, up, args.warmup, args.repeats),
            timed(b, gate, up, args.warmup, args.repeats),
            timed(a, gate, up, args.warmup, args.repeats),
        ]
        incumbent_us = (values[0] + values[3]) / 2.0
        candidate_us = (values[1] + values[2]) / 2.0
        rows[f"m{m}_d{d}"] = {
            "A1_us": values[0],
            "B1_us": values[1],
            "B2_us": values[2],
            "A2_us": values[3],
            "incumbent_mean_us": incumbent_us,
            "candidate_mean_us": candidate_us,
            "candidate_over_incumbent_speedup": incumbent_us / candidate_us,
            "max_abs": max_abs,
            "max_rel": max_rel,
        }

    speedups = [row["candidate_over_incumbent_speedup"] for row in rows.values()]
    payload = {
        "schema_version": 1,
        "protocol": "same_process_ABBA",
        "order": "A1,B1,B2,A2",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "shapes": rows,
        "arithmetic_mean_speedup": sum(speedups) / len(speedups),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
