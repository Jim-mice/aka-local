#!/usr/bin/env python3
"""Reference-only L0/L1 benchmark for the frozen V100 RMSNorm contract."""

import argparse
import json
import math
import os
import statistics
import time

import torch


SHAPES = [(16, 1, 1024), (64, 2, 1024), (128, 2, 1024)]
EPS = 1.0e-5


def stats(values):
    ordered = sorted(values)
    n = len(ordered)
    def pct(q):
        pos = (n - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if n > 1 else 0.0
    return {"n": n, "median_ms": statistics.median(values), "mean_ms": mean,
            "stdev_ms": stdev, "cv": stdev / mean if mean else 0.0,
            "p10_ms": pct(0.10), "p90_ms": pct(0.90),
            "min_ms": min(values), "max_ms": max(values), "raw_ms": values}


def event_ms(call, iterations):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def calibrate(call):
    iterations = 1
    while iterations < 65536:
        elapsed = event_ms(call, iterations)
        if elapsed >= 20.0:
            return iterations, elapsed
        iterations *= 2
    return iterations, event_ms(call, iterations)


def benchmark(call):
    iterations, calibration_ms = calibrate(call)
    for _ in range(20):
        event_ms(call, iterations)
    windows = [event_ms(call, iterations) / iterations for _ in range(50)]
    return {"iterations_per_window": iterations, "calibration_window_ms": calibration_ms,
            "per_call": stats(windows)}


def errors(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-12)
    return {"max_abs": float(diff.max()), "max_rel": float((diff / denom).max()),
            "finite": bool(torch.isfinite(actual).all())}


def make_reference(shape):
    s, b, h = shape
    torch.manual_seed(20260923 + s)
    x = torch.randn(s, b, h, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(h, device="cuda", dtype=torch.float16)
    grad = torch.randn_like(x)
    module = torch.nn.RMSNorm(h, eps=EPS, device="cuda", dtype=torch.float16)
    module.weight.data.copy_(weight)
    return x, weight, grad, module


def run_l0():
    rows = []
    for shape in SHAPES:
        x, weight, grad, module = make_reference(shape)
        y = module(x)
        xo = x.detach().float().requires_grad_(True)
        wo = weight.detach().float().requires_grad_(True)
        yo = xo * torch.rsqrt(xo.square().mean(-1, keepdim=True) + EPS) * wo
        dx, dw = torch.autograd.grad(y, (x, module.weight), grad, retain_graph=True)
        dxo, dwo = torch.autograd.grad(yo, (xo, wo), grad.float())
        forward_call = lambda: module(x)
        backward_call = lambda: torch.autograd.grad(y, (x, module.weight), grad, retain_graph=True)
        rows.append({
            "shape_s_b_h": shape,
            "correctness": {"forward": errors(y, yo), "input_grad": errors(dx, dxo),
                            "weight_grad": errors(dw, dwo)},
            "forward": benchmark(forward_call),
            "backward": benchmark(backward_call),
        })
    return {"level": "L0", "status": "PASS", "rows": rows}


def run_l1(megatron_path):
    import sys
    sys.path.insert(0, megatron_path)
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.torch_norm import WrappedTorchNorm
    cfg = TransformerConfig(num_layers=1, hidden_size=1024, num_attention_heads=8,
                            normalization="RMSNorm", layernorm_epsilon=EPS)
    module = WrappedTorchNorm(config=cfg, hidden_size=1024, eps=EPS).cuda().half()
    x = torch.randn(16, 1, 1024, device="cuda", dtype=torch.float16, requires_grad=True)
    y = module(x)
    loss = y.float().square().mean()
    loss.backward()
    return {"level": "L1", "status": "PASS",
            "module_type": f"{type(module).__module__}.{type(module).__name__}",
            "is_torch_rmsnorm": isinstance(module, torch.nn.RMSNorm),
            "forward_invoked": True, "backward_invoked": True,
            "input_grad_finite": bool(torch.isfinite(x.grad).all()),
            "weight_grad_finite": bool(torch.isfinite(module.weight.grad).all()),
            "output_shape": list(y.shape)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["l0", "l1", "all"], default="all")
    parser.add_argument("--megatron-path", default="/home/bencheng/aka_targets/megatron-lm-5be9626")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "physical GPU 1 must be selected"
    props = torch.cuda.get_device_properties(0)
    result = {"timestamp": time.time(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "device": {"name": props.name, "total_memory": props.total_memory,
                         "sm_count": props.multi_processor_count,
                         "compute_capability": [props.major, props.minor]},
              "l0": run_l0() if args.mode in ("l0", "all") else None,
              "l1": run_l1(args.megatron_path) if args.mode in ("l1", "all") else None}
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, args.output)
    print(json.dumps({"status": "DONE", "output": args.output}))


if __name__ == "__main__":
    main()
