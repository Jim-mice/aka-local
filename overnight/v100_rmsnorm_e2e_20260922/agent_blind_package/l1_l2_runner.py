#!/usr/bin/env python3
"""Controlled Megatron GPTModel L1/L2 evaluator for the frozen contract."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("current_agent_candidate_l2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def replace_rmsnorms(module, candidate_module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.RMSNorm):
            hidden = child.normalized_shape[0] if isinstance(child.normalized_shape, tuple) else child.normalized_shape
            replacement = candidate_module.TritonRMSNorm(hidden, eps=child.eps, num_warps=4).cuda().half()
            replacement.weight.data.copy_(child.weight.data)
            setattr(module, name, replacement)
            count += 1
        else:
            count += replace_rmsnorms(child, candidate_module)
    return count


def summary(values):
    ordered = sorted(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    def pct(q):
        p = (len(ordered) - 1) * q
        lo, hi = math.floor(p), math.ceil(p)
        return ordered[lo] if lo == hi else ordered[lo] * (hi - p) + ordered[hi] * (p - lo)
    return {"n": len(values), "median_ms": statistics.median(values), "mean_ms": mean,
            "stdev_ms": sd, "cv": sd / mean if mean else 0.0, "p10_ms": pct(.1),
            "p90_ms": pct(.9), "min_ms": min(values), "max_ms": max(values), "raw_ms": values}


def timed_step(call, iterations):
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    host_start = time.perf_counter()
    start_event.record()
    for _ in range(iterations):
        call()
    end_event.record()
    end_event.synchronize()
    host_ms = (time.perf_counter() - host_start) * 1000.0
    return start_event.elapsed_time(end_event), host_ms


def calibrate(call):
    iterations = 1
    while iterations < 64:
        event_ms, _ = timed_step(call, iterations)
        if event_ms >= 20.0:
            return iterations
        iterations *= 2
    return iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--megatron-path", default="/home/bencheng/aka_targets/megatron-lm-5be9626")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    import sys
    sys.path.insert(0, args.megatron_path)
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{free_port()}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(20260923)
    torch.manual_seed(20260923)
    torch.cuda.manual_seed_all(20260923)
    config = TransformerConfig(
        num_layers=2, hidden_size=1024, num_attention_heads=8, ffn_hidden_size=4096,
        normalization="RMSNorm", layernorm_epsilon=1.0e-5, params_dtype=torch.float16,
        fp16=True, use_cpu_initialization=True, hidden_dropout=0.0, attention_dropout=0.0,
        masked_softmax_fusion=False, bias_activation_fusion=False, bias_dropout_fusion=False,
    )
    spec = get_gpt_layer_local_spec()
    build = lambda: GPTModel(config=config, transformer_layer_spec=spec, vocab_size=2048,
                             max_sequence_length=128, position_embedding_type="rope",
                             parallel_output=False).cuda().half()
    reference = build()
    candidate = build()
    candidate.load_state_dict(reference.state_dict())
    candidate_module = load_candidate(args.candidate)
    reference_norms = sum(isinstance(m, torch.nn.RMSNorm) for m in reference.modules())
    replaced = replace_rmsnorms(candidate, candidate_module)
    remaining_reference_impls = sum(isinstance(m, torch.nn.RMSNorm) for m in candidate.modules())
    candidate_norms = [m for m in candidate.modules() if isinstance(m, candidate_module.TritonRMSNorm)]

    tokens = torch.randint(0, 2048, (2, 128), device="cuda", dtype=torch.long)
    positions = torch.arange(128, device="cuda").unsqueeze(0).expand(2, 128)
    attention_mask = torch.triu(torch.ones(128, 128, device="cuda", dtype=torch.bool), diagonal=1)
    attention_mask = attention_mask.view(1, 1, 128, 128).expand(2, 1, 128, 128)
    labels = tokens.clone()
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.0)
    cand_opt = torch.optim.SGD(candidate.parameters(), lr=0.0)

    def ref_step():
        ref_opt.zero_grad(set_to_none=True)
        loss = reference(input_ids=tokens, position_ids=positions, attention_mask=attention_mask, labels=labels).float().mean()
        loss.backward()
        ref_opt.step()
        return loss

    def cand_step():
        cand_opt.zero_grad(set_to_none=True)
        loss = candidate(input_ids=tokens, position_ids=positions, attention_mask=attention_mask, labels=labels).float().mean()
        loss.backward()
        cand_opt.step()
        return loss

    ref_loss = ref_step().detach()
    cand_loss = cand_step().detach()
    loss_abs = float((ref_loss - cand_loss).abs())
    grad_pairs = []
    for (rn, rp), (cn, cp) in zip(reference.named_parameters(), candidate.named_parameters()):
        if rp.grad is not None and cp.grad is not None and ("layer_norm" in rn or "layernorm" in rn):
            grad_pairs.append(float((rp.grad.float() - cp.grad.float()).abs().max()))
    l1 = {"status": "PASS" if replaced == reference_norms and remaining_reference_impls == 0 else "FAIL",
          "reference_rmsnorm_modules": reference_norms, "candidate_replacements": replaced,
          "remaining_torch_rmsnorm_in_candidate": remaining_reference_impls,
          "replacement_invocations": sum(m.invocation_count for m in candidate_norms),
          "forward": True, "backward": True,
          "input_and_parameter_gradients_finite": all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters()),
          "loss_abs_error": loss_abs, "norm_grad_max_abs_errors": grad_pairs}

    if args.smoke:
        l2 = {"status": "SMOKE_PASS", "reference_loss": float(ref_loss), "candidate_loss": float(cand_loss)}
    else:
        iterations = max(calibrate(ref_step), calibrate(cand_step))
        for i in range(20):
            (ref_step if i % 2 == 0 else cand_step)()
        ref_gpu, cand_gpu, ref_host, cand_host = [], [], [], []
        for cycle in range(25):
            sequence = ("A", "B", "B", "A") if cycle % 2 == 0 else ("B", "A", "A", "B")
            for label in sequence:
                gpu_ms, host_ms = timed_step(ref_step if label == "A" else cand_step, iterations)
                target_gpu, target_host = (ref_gpu, ref_host) if label == "A" else (cand_gpu, cand_host)
                target_gpu.append(gpu_ms / iterations)
                target_host.append(host_ms / iterations)
        n = min(len(ref_gpu), len(cand_gpu))
        deltas = [cand_gpu[i] - ref_gpu[i] for i in range(n)]
        l2 = {"status": "PASS", "name": "CONTROLLED_MEGATRON_E2E", "iterations_per_window": iterations,
              "reference_gpu": summary(ref_gpu), "candidate_gpu": summary(cand_gpu),
              "reference_host": summary(ref_host), "candidate_host": summary(cand_host),
              "paired_delta_gpu_ms": summary(deltas),
              "speedup_gpu_mean": statistics.fmean(ref_gpu) / statistics.fmean(cand_gpu),
              "config": {"layers":2,"S":128,"B":2,"H":1024,"heads":8,"ffn_hidden":4096,"vocab":2048,"optimizer":"SGD lr=0.0"}}

    result = {"status": "PASS" if l1["status"] == "PASS" else "FAIL", "l1": l1, "l2": l2,
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "candidate_sha256": hashlib.sha256(open(args.candidate, "rb").read()).hexdigest()}
    tmp = args.output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, args.output)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    print(json.dumps({"status": result["status"], "output": args.output}))


if __name__ == "__main__":
    main()
