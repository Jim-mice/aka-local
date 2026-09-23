#!/usr/bin/env python3
"""Same-process interleaved evaluator for CURRENT-AGENT RMSNorm candidates."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import time

import torch


EPS = 1.0e-5
ALL_SHAPES = [(16, 1, 1024), (64, 2, 1024), (128, 2, 1024)]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("current_agent_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(values):
    ordered = sorted(values)
    n = len(ordered)
    def percentile(q):
        p = (n - 1) * q
        lo, hi = math.floor(p), math.ceil(p)
        return ordered[lo] if lo == hi else ordered[lo] * (hi - p) + ordered[hi] * (p - lo)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return {"n": n, "median_ms": statistics.median(values), "mean_ms": mean,
            "stdev_ms": sd, "cv": sd / mean if mean else 0.0,
            "p10_ms": percentile(0.1), "p90_ms": percentile(0.9),
            "min_ms": min(values), "max_ms": max(values), "raw_ms": values}


def elapsed(call, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def calibration(call, target_ms):
    iterations = 1
    while iterations < 65536:
        value = elapsed(call, iterations)
        if value >= target_ms:
            return iterations
        iterations *= 2
    return iterations


def paired_benchmark(ref_call, cand_call, quick):
    target_ms = 2.0 if quick else 20.0
    iterations = max(calibration(ref_call, target_ms), calibration(cand_call, target_ms))
    warmup = 5 if quick else 20
    for i in range(warmup):
        (ref_call if i % 2 == 0 else cand_call)()
    torch.cuda.synchronize()
    sequence_a = ("A", "B", "B", "A")
    sequence_b = ("B", "A", "A", "B")
    cycles = 3 if quick else 25
    a_values, b_values = [], []
    for cycle in range(cycles):
        for label in (sequence_a if cycle % 2 == 0 else sequence_b):
            value = elapsed(ref_call if label == "A" else cand_call, iterations) / iterations
            (a_values if label == "A" else b_values).append(value)
    count = min(len(a_values), len(b_values))
    if not quick and (len(a_values) < 50 or len(b_values) < 50):
        raise RuntimeError("BENCHMARK_PROTOCOL_FAILURE: official measured windows below 50 per side")
    deltas = [b_values[i] - a_values[i] for i in range(count)]
    return {"iterations_per_window": iterations, "reference": summarize(a_values),
            "candidate": summarize(b_values), "paired_delta_ms": summarize(deltas),
            "speedup_mean": statistics.fmean(a_values) / statistics.fmean(b_values)}


def tensor_error(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    return {"max_abs": float(diff.max()),
            "max_rel": float((diff / expected.float().abs().clamp_min(1e-12)).max()),
            "finite": bool(torch.isfinite(actual).all()),
            "allclose": bool(torch.allclose(actual.float(), expected.float(), atol=0.005, rtol=0.005))}


def evaluate_shape(candidate_module, shape, num_warps, quick):
    s, b, h = shape
    torch.manual_seed(20260923 + s)
    base = torch.randn(s, b, h, device="cuda", dtype=torch.float16)
    weight = torch.randn(h, device="cuda", dtype=torch.float16)
    grad = torch.randn_like(base)
    xr = base.detach().requires_grad_(True)
    xc = base.detach().requires_grad_(True)
    ref = torch.nn.RMSNorm(h, eps=EPS, device="cuda", dtype=torch.float16)
    cand = candidate_module.TritonRMSNorm(h, eps=EPS, num_warps=num_warps).cuda().half()
    ref.weight.data.copy_(weight)
    cand.weight.data.copy_(weight)
    yr, yc = ref(xr), cand(xc)
    dxr, dwr = torch.autograd.grad(yr, (xr, ref.weight), grad, retain_graph=True)
    dxc, dwc = torch.autograd.grad(yc, (xc, cand.weight), grad, retain_graph=True)
    correctness = {"forward": tensor_error(yc, yr), "input_grad": tensor_error(dxc, dxr),
                   "weight_grad": tensor_error(dwc, dwr)}
    forward = paired_benchmark(lambda: ref(xr), lambda: cand(xc), quick)
    backward = paired_benchmark(
        lambda: torch.autograd.grad(yr, (xr, ref.weight), grad, retain_graph=True),
        lambda: torch.autograd.grad(yc, (xc, cand.weight), grad, retain_graph=True), quick)
    return {"shape_s_b_h": shape, "num_warps": num_warps, "correctness": correctness,
            "correctness_pass": all(v["finite"] and v["allclose"] for v in correctness.values()),
            "forward": forward, "backward": backward,
            "candidate_invocations": cand.invocation_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mode", choices=["quick", "official"], default="quick")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    quick = args.mode == "quick"
    shapes = [ALL_SHAPES[0], ALL_SHAPES[-1]] if quick else ALL_SHAPES
    candidate_module = load_candidate(args.candidate)
    variants = [4, 8] if quick else [4]
    rows = []
    status = "PASS"
    try:
        for num_warps in variants:
            for shape in shapes:
                row = evaluate_shape(candidate_module, shape, num_warps, quick)
                rows.append(row)
                if not row["correctness_pass"]:
                    status = "CORRECTNESS_FAILURE"
    except Exception as exc:
        status = "COMPILE_FAILURE" if not rows else "FRAMEWORK_FAILURE"
        rows.append({"exception_type": type(exc).__name__, "exception": str(exc)})
    if not quick and any(row.get("correctness_pass") is False for row in rows):
        status = "CORRECTNESS_FAILURE"
    result = {"status": status, "mode": args.mode, "timestamp_epoch": time.time(),
              "candidate_sha256": file_hash(args.candidate), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "rows": rows}
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, args.output)
    print(json.dumps({"status": status, "output": args.output}))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
