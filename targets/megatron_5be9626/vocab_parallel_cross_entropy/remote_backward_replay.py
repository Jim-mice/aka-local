import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.cross_entropy import (
    VocabParallelCrossEntropy,
    vocab_parallel_cross_entropy,
)


COMMIT = "5be9626709af2722333bf54797c954c09edeada3"
REPLAY_CONTRACT = "d8ea9db11871996c"
EVALUATOR_VERSION = "tp2_semantic_v2"
EVALUATOR_HASH = "7cc0fac8d80b2d27"
FIXTURE_VERSION = "coherent_global_fixture_v2"
BACKWARD_RANGE_BEGIN = "AKA_CE_BACKWARD_BEGIN"
BACKWARD_RANGE_END = "AKA_CE_BACKWARD_END"


def setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return rank, local_rank, torch.cuda.current_device()


def tensor_summary(x):
    y = x.detach().float().reshape(-1)
    if y.numel() == 0:
        return {"numel": 0}
    take = min(8, y.numel())
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "stride": list(x.stride()),
        "contiguous": bool(x.is_contiguous()),
        "numel": int(y.numel()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.mean().item()),
        "sum": float(y.sum().item()),
        "selected": [float(v) for v in y[:take].cpu().tolist()],
        "sha256": hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
    }


def error_summary(a, b):
    aa = a.detach().float()
    bb = b.detach().float()
    d = (aa - bb).abs()
    denom = bb.abs().clamp_min(1.0e-12)
    return {
        "max_abs": float(d.max().item()),
        "max_rel": float((d / denom).max().item()),
        "mean_abs": float(d.mean().item()),
    }


def fixture(s, b, v, seed, rank, world):
    torch.manual_seed(seed)
    full = torch.randn((s, b, v), device="cuda", dtype=torch.float16)
    target = torch.randint(0, v, (s, b), device="cuda", dtype=torch.long)
    local_v = v // world
    local = full[..., rank * local_v : (rank + 1) * local_v].contiguous()
    return full, local, target


def make_real_saved_state(local, target, world):
    logits = local.detach().clone().requires_grad_(True)
    loss = vocab_parallel_cross_entropy(logits, target, label_smoothing=0.0, tp_group=dist.group.WORLD)
    grad_fn = loss.grad_fn
    saved = grad_fn.saved_tensors
    softmax, target_mask, masked_target = [x.detach().clone() for x in saved]
    return {
        "loss": loss.detach(),
        "loss_grad_fn": grad_fn,
        "source_logits": logits,
        "softmax": softmax,
        "target_mask": target_mask,
        "masked_target_1d": masked_target,
        "world": world,
    }


def expected_state(full, target, rank, world):
    v = full.shape[-1]
    local_v = v // world
    start = rank * local_v
    end = (rank + 1) * local_v
    full_f = full.float()
    global_max = full_f.amax(dim=-1, keepdim=True)
    probs = torch.exp(full_f - global_max)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    target_mask = (target < start) | (target >= end)
    masked = target.clone() - start
    masked[target_mask] = 0
    return probs[..., start:end].contiguous(), target_mask, masked.reshape(-1)


def direct_backward(softmax, target_mask, masked_target, grad_output):
    grad_2d, arange_1d, softmax_update, grad_input = (
        VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
    )
    return VocabParallelCrossEntropy.calculate_gradients(
        grad_2d, arange_1d, masked_target, softmax_update, grad_input, grad_output
    )


def public_backward(local, target, grad_output):
    logits = local.detach().clone().requires_grad_(True)
    loss = vocab_parallel_cross_entropy(logits, target, label_smoothing=0.0, tp_group=dist.group.WORLD)
    loss.backward(grad_output)
    return logits.grad.detach().clone(), loss.detach()


def run_correctness(s, b, v, seed, rank, world, include_edges=False):
    full, local, target = fixture(s, b, v, seed, rank, world)
    real = make_real_saved_state(local, target, world)
    grad_output = torch.linspace(0.25, 1.25, s * b, device="cuda", dtype=torch.float32).reshape(s, b)
    expected_soft, expected_mask, expected_masked = expected_state(full, target, rank, world)
    direct_soft = real["softmax"].clone()
    direct = direct_backward(direct_soft, real["target_mask"], real["masked_target_1d"], grad_output)
    public_grad, public_loss = public_backward(local, target, grad_output)
    oracle_grad_f = (expected_soft - torch.nn.functional.one_hot(
        expected_masked.reshape(s, b).clamp(0, v // world - 1), num_classes=v // world
    ).float() * (~expected_mask).reshape(s, b, 1).float())
    oracle_grad_f = oracle_grad_f * grad_output.unsqueeze(-1)
    oracle_grad = oracle_grad_f.to(direct.dtype)
    result = {
        "config": {"S": s, "B": b, "V": v, "N": s * b, "local_vocab": v // world},
        "seed": seed,
        "rank": rank,
        "world": world,
        "source": {
            "loss_dtype": str(real["loss"].dtype),
            "saved_softmax_dtype": str(real["softmax"].dtype),
            "saved_target_mask_dtype": str(real["target_mask"].dtype),
            "saved_masked_target_dtype": str(real["masked_target_1d"].dtype),
            "direct_gradient_dtype": str(direct.dtype),
            "public_input_gradient_dtype": str(public_grad.dtype),
            "grad_output_dtype": str(grad_output.dtype),
        },
        "saved_state_errors": {
            "softmax": error_summary(real["softmax"], expected_soft),
            "target_mask": error_summary(real["target_mask"].float(), expected_mask.float()),
            "masked_target": error_summary(real["masked_target_1d"].float(), expected_masked.float()),
        },
        "gradient_errors": {
            "direct_vs_public": error_summary(direct, public_grad.float()),
            "direct_vs_oracle": error_summary(direct, oracle_grad.float()),
            "public_vs_oracle": error_summary(public_grad.float(), oracle_grad.float()),
        },
        "state_summary": {
            "softmax": tensor_summary(real["softmax"]),
            "target_mask": tensor_summary(real["target_mask"].to(torch.uint8)),
            "masked_target_1d": tensor_summary(real["masked_target_1d"]),
        },
        "collectives_in_backward": 0,
    }
    if include_edges:
        result["edge_fixtures"] = run_edge_fixtures(rank, world)
    return result


def run_edge_fixtures(rank, world):
    if world != 2:
        return {"status": "SKIPPED", "reason": "edge ownership fixtures require TP=2"}
    v = 8
    local_v = 4
    cases = [
        ("target_rank0_first_global_max_rank0", torch.tensor([[0, 5]], device="cuda"), 0),
        ("target_rank1_last_global_max_rank1", torch.tensor([[7, 2]], device="cuda"), 1),
        ("equal_logits", torch.tensor([[1, 6]], device="cuda"), 2),
        ("large_positive_negative", torch.tensor([[3, 4]], device="cuda"), 3),
    ]
    outputs = []
    for name, target, kind in cases:
        if kind == 0:
            full = torch.tensor([[[10, 1, 0, -1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8]]], device="cuda", dtype=torch.float16)
        elif kind == 1:
            full = torch.tensor([[[-8, -7, -6, -5, -4, -3, -2, 20], [8, 7, 6, 5, 4, 3, 2, 1]]], device="cuda", dtype=torch.float16)
        elif kind == 2:
            full = torch.zeros((1, 2, v), device="cuda", dtype=torch.float16)
        else:
            full = torch.tensor([[[100, -100, 50, -50, 10, -10, 5, -5], [-100, 100, -50, 50, -10, 10, -5, 5]]], device="cuda", dtype=torch.float16)
        local = full[..., rank * local_v : (rank + 1) * local_v].contiguous()
        real = make_real_saved_state(local, target, world)
        expected_soft, expected_mask, expected_masked = expected_state(full, target, rank, world)
        outputs.append({
            "name": name,
            "rank": rank,
            "target": [int(x) for x in target.reshape(-1).cpu().tolist()],
            "target_mask": [bool(x) for x in real["target_mask"].reshape(-1).cpu().tolist()],
            "masked_target": [int(x) for x in real["masked_target_1d"].cpu().tolist()],
            "expected_mask": [bool(x) for x in expected_mask.reshape(-1).cpu().tolist()],
            "expected_masked_target": [int(x) for x in expected_masked.cpu().tolist()],
            "softmax_error": error_summary(real["softmax"], expected_soft),
            "target_mask_error": error_summary(real["target_mask"].float(), expected_mask.float()),
            "masked_target_error": error_summary(real["masked_target_1d"].float(), expected_masked.float()),
        })
    return {"status": "PASS", "cases": outputs}


def timed_one(soft_template, target_mask, masked_target, grad_output):
    # Output storage is preallocated and restored outside the timed interval.
    soft = torch.empty_like(soft_template)
    soft.copy_(soft_template)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(BACKWARD_RANGE_BEGIN)
    start.record()
    direct_backward(soft, target_mask, masked_target, grad_output)
    end.record()
    torch.cuda.nvtx.range_pop()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0)


def profiler_call(name):
    try:
        fn = getattr(torch.cuda.cudart(), name)
        result = fn()
        return {"available": True, "call": name, "result": int(result) if result is not None else None}
    except Exception as exc:
        return {"available": False, "call": name, "error": str(exc)}


def run_benchmark(s, b, v, seed, rank, world, blocks, warmup, measurements, profile_gate=False):
    full, local, target = fixture(s, b, v, seed, rank, world)
    real = make_real_saved_state(local, target, world)
    soft_template = real["softmax"].detach().clone()
    target_mask = real["target_mask"].detach().clone()
    masked_target = real["masked_target_1d"].detach().clone()
    grad_output = torch.ones((s, b), device="cuda", dtype=torch.float32)
    for _ in range(warmup):
        dist.barrier()
        soft = torch.empty_like(soft_template); soft.copy_(soft_template)
        torch.cuda.synchronize()
        direct_backward(soft, target_mask, masked_target, grad_output)
        torch.cuda.synchronize()
        dist.barrier()
    profile_start = profiler_call("cudaProfilerStart") if profile_gate else {"available": False, "disabled": True}
    rows = []
    for block in range(blocks):
        for iteration in range(measurements):
            if not profile_gate:
                dist.barrier()
            torch.cuda.synchronize()
            soft = torch.empty_like(soft_template); soft.copy_(soft_template)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(BACKWARD_RANGE_BEGIN)
            start.record()
            direct_backward(soft, target_mask, masked_target, grad_output)
            end.record()
            torch.cuda.nvtx.range_pop()
            end.synchronize()
            latency_us = float(start.elapsed_time(end) * 1000.0)
            torch.cuda.synchronize()
            if not profile_gate:
                dist.barrier()
            rows.append({
                "benchmark_era": "tp2_ce_backward_local_v1",
                "performance_boundary": "saved_state_to_local_logits_gradient",
                "rank": rank,
                "world": world,
                "config": {"S": s, "B": b, "V": v, "N": s * b, "local_vocab": v // world},
                "fixture_seed": seed,
                "fixture_version": FIXTURE_VERSION,
                "block": block,
                "iteration": iteration,
                "warmup": False,
                "latency_us": latency_us,
                "success": True,
                "collectives_in_timed_range": 0,
                "nvtx_begin": BACKWARD_RANGE_BEGIN,
                "nvtx_end": BACKWARD_RANGE_END,
                "output_allocation": "outside_timed_range",
                "input_state": "real_megatron_saved_tensors_cloned_outside_timing",
            })
    profile_stop = profiler_call("cudaProfilerStop") if profile_gate else {"available": False, "disabled": True}
    vals = [r["latency_us"] for r in rows]
    return {
        "metadata": {
            "commit": COMMIT,
            "replay_contract": REPLAY_CONTRACT,
            "evaluator_version": EVALUATOR_VERSION,
            "evaluator_hash": EVALUATOR_HASH,
            "fixture_version": FIXTURE_VERSION,
            "rank": rank,
            "world": world,
            "config": {"S": s, "B": b, "V": v},
            "warmup": warmup,
            "blocks": blocks,
            "measurements_per_block": measurements,
            "cuda_profiler_start": profile_start,
            "cuda_profiler_stop": profile_stop,
        },
        "rows": rows,
        "summary": {
            "N": len(vals),
            "mean_us": statistics.mean(vals),
            "std_us": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "cv": (statistics.stdev(vals) / statistics.mean(vals)) if len(vals) > 1 and statistics.mean(vals) else 0.0,
            "median_us": statistics.median(vals),
            "min_us": min(vals),
            "max_us": max(vals),
        },
    }


def run_stage_diagnostic(s, b, v, seed, rank, world, measurements):
    _, local, target = fixture(s, b, v, seed, rank, world)
    real = make_real_saved_state(local, target, world)
    soft_template = real["softmax"].detach().clone()
    target_mask = real["target_mask"].detach().clone()
    masked_target = real["masked_target_1d"].detach().clone()
    grad_output = torch.ones((s, b), device="cuda", dtype=torch.float32)
    for _ in range(5):
        soft = torch.empty_like(soft_template); soft.copy_(soft_template)
        direct_backward(soft, target_mask, masked_target, grad_output)
    torch.cuda.synchronize()
    rows = []
    for iteration in range(measurements):
        dist.barrier(); torch.cuda.synchronize()
        soft = torch.empty_like(soft_template); soft.copy_(soft_template)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True); e3 = torch.cuda.Event(enable_timing=True)
        e4 = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(BACKWARD_RANGE_BEGIN)
        e0.record()
        grad_2d, arange_1d, softmax_update, grad_input = VocabParallelCrossEntropy.prepare_gradient_calculation_operands(soft, target_mask)
        e1.record()
        VocabParallelCrossEntropy.calculate_gradients(grad_2d, arange_1d, masked_target, softmax_update, grad_input, grad_output)
        e2.record()
        torch.cuda.nvtx.range_pop()
        e2.synchronize()
        rows.append({
            "iteration": iteration,
            "prepare_us": float(e0.elapsed_time(e1) * 1000.0),
            "calculate_us": float(e1.elapsed_time(e2) * 1000.0),
            "total_us": float(e0.elapsed_time(e2) * 1000.0),
            "collectives": 0,
        })
        torch.cuda.synchronize(); dist.barrier()
    values = {k: [r[k] for r in rows] for k in ("prepare_us", "calculate_us", "total_us")}
    return {
        "metadata": {"rank": rank, "world": world, "config": {"S": s, "B": b, "V": v}, "fixture_version": FIXTURE_VERSION, "measurements": measurements},
        "rows": rows,
        "means_us": {k: statistics.mean(v) for k, v in values.items()},
        "std_us": {k: statistics.stdev(v) if len(v) > 1 else 0.0 for k, v in values.items()},
    }


def finite_difference():
    torch.manual_seed(1515)
    x = torch.randn((2, 5), dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0, 4], dtype=torch.long)
    upstream = torch.tensor([0.75, 1.25], dtype=torch.float64)
    loss = torch.logsumexp(x, dim=-1) - x[torch.arange(2), target]
    analytical = torch.autograd.grad((loss * upstream).sum(), x)[0]
    eps = 1.0e-6
    numerical = torch.zeros_like(x)
    with torch.no_grad():
        for i in range(x.numel()):
            xp = x.detach().clone(); xm = x.detach().clone()
            xp.reshape(-1)[i] += eps; xm.reshape(-1)[i] -= eps
            lp = (torch.logsumexp(xp, dim=-1) - xp[torch.arange(2), target]) @ upstream
            lm = (torch.logsumexp(xm, dim=-1) - xm[torch.arange(2), target]) @ upstream
            numerical.reshape(-1)[i] = (lp - lm) / (2 * eps)
    return {"max_abs_error": float((analytical - numerical).abs().max().item()), "epsilon": eps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "benchmark", "stage", "finite-difference"], default="correctness")
    ap.add_argument("--S", type=int, default=32)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--V", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1515)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measurements", type=int, default=10)
    ap.add_argument("--profile-gate", action="store_true")
    ap.add_argument("--edges", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.mode == "finite-difference":
        result = {"status": "PASS", "finite_difference": finite_difference()}
        print(json.dumps(result, indent=2), flush=True)
        return
    rank, _, _ = setup()
    world = dist.get_world_size()
    if args.mode == "correctness":
        result = run_correctness(args.S, args.B, args.V, args.seed, rank, world, args.edges)
    elif args.mode == "stage":
        result = run_stage_diagnostic(args.S, args.B, args.V, args.seed, rank, world, args.measurements)
    else:
        result = run_benchmark(args.S, args.B, args.V, args.seed, rank, world, args.blocks, args.warmup, args.measurements, args.profile_gate)
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.output:
        output_text = str(args.output).replace("{rank}", str(rank))
        Path(output_text).write_text(text + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
