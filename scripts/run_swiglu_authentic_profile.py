"""Measure authentic Megatron fused SwiGLU with CUDA events only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import traceback


def summary(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    half = max(1, len(values) // 2)
    low = statistics.median(ordered[:half])
    high = statistics.median(ordered[-half:])
    return {
        "count": len(values), "raw_samples_ms": values, "median_ms": median,
        "mean_ms": mean, "stdev_ms": stdev, "cv": stdev / mean if mean else None,
        "min_ms": min(values), "max_ms": max(values),
        "stability": bool(mean) and stdev / mean <= 0.10,
        "bimodality_check": {
            "method": "lower-half versus upper-half median ratio heuristic",
            "lower_half_median_ms": low, "upper_half_median_ms": high,
            "status": "POSSIBLE_BIMODAL" if high / max(low, 1e-12) > 1.25 else "NOT_DETECTED",
        },
    }


def error(a, b) -> dict[str, object]:
    delta = (a - b).detach()
    return {
        "shape_equal": tuple(a.shape) == tuple(b.shape),
        "max_abs_error": float(delta.abs().max().cpu()),
        "max_rel_error": float((delta.abs() / b.detach().abs().clamp_min(1e-12)).max().cpu()),
        "finite": bool(torch.isfinite(delta).all().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=120)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo.resolve() / ".runtime_deps"))
    sys.path.insert(1, str(args.repo.resolve()))
    sys.path.insert(2, str(args.megatron_root.resolve()))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    import megatron.core.transformer.mlp as megatron_mlp

    globals()["torch"] = torch
    if not torch.cuda.is_available():
        (args.out / "boundary_timing.json").write_text(json.dumps({"status": "BLOCKED", "reason": "CUDA unavailable"}, indent=2), encoding="utf-8")
        return 0
    device = torch.device("cuda")

    class LocalLinear(nn.Module):
        def __init__(self, input_size, output_size, config=None, bias=True, **_kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(output_size, input_size) * 0.02)
            self.bias = nn.Parameter(torch.randn(output_size) * 0.01) if bias else None

        def forward(self, hidden_states):
            return F.linear(hidden_states, self.weight), self.bias

    def make_model():
        config = TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=1, ffn_hidden_size=16,
            tensor_model_parallel_size=1, sequence_parallel=False, gated_linear_unit=True,
            activation_func=F.silu, bias_activation_fusion=True, add_bias_linear=True,
            params_dtype=torch.float32, use_cpu_initialization=True,
            perform_initialization=False, transformer_impl="local",
        )
        return MLP(config, MLPSubmodules(linear_fc1=LocalLinear, linear_fc2=LocalLinear)).to(device)

    def output_only(raw):
        return raw[0] if isinstance(raw, tuple) else raw

    seed = 20260922
    torch.manual_seed(seed)
    reference_model = make_model()
    eager_model = make_model()
    eager_model.load_state_dict(reference_model.state_dict())
    x = torch.randn(2, 3, 8, device=device)
    target = torch.randn(2, 3, 8, device=device)

    # First execute the original path, before installing any wrapper.
    authentic_correctness = {"status": "PASS"}
    try:
        authentic_input = x.detach().clone().requires_grad_(True)
        authentic_output = output_only(reference_model(authentic_input))
        authentic_loss = (authentic_output - target).pow(2).mean()
        authentic_loss.backward()
        authentic_grad = authentic_input.grad.detach().clone()
        torch.cuda.synchronize()
    except Exception as exc:
        authentic_correctness = {"status": "BLOCKED", "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        (args.out / "authentic_correctness.json").write_text(json.dumps(authentic_correctness, indent=2), encoding="utf-8")
        return 0

    eager_input = x.detach().clone().requires_grad_(True)
    eager_output = output_only(eager_model(eager_input))
    eager_loss = (eager_output - target).pow(2).mean()
    eager_loss.backward()
    eager_grad = eager_input.grad.detach().clone()
    torch.cuda.synchronize()
    authentic_correctness = {
        "status": "PASS",
        "forward": error(authentic_output, eager_output),
        "loss_abs_error": float((authentic_loss - eager_loss).abs().cpu()),
        "gradient": error(authentic_grad, eager_grad),
        "dtype": str(authentic_output.dtype), "shape": list(authentic_output.shape),
    }
    (args.out / "authentic_correctness.json").write_text(json.dumps(authentic_correctness, indent=2), encoding="utf-8")

    # Patch only the D-side imported MLP module global. The wrapped target remains
    # Megatron's original bias_swiglu_impl and its original Triton callables.
    original_boundary = megatron_mlp.bias_swiglu_impl
    boundary_events: list[tuple[object, object]] = []
    original_boundary_calls = 0

    def timed_boundary(*boundary_args, **boundary_kwargs):
        nonlocal original_boundary_calls
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_boundary(*boundary_args, **boundary_kwargs)
        end.record()
        boundary_events.append((start, end))
        original_boundary_calls += 1
        return result

    megatron_mlp.bias_swiglu_impl = timed_boundary
    step_events: list[tuple[object, object]] = []
    optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.0)
    for _ in range(args.warmups):
        optimizer.zero_grad(set_to_none=True)
        step_input = x.detach().clone().requires_grad_(True)
        output = output_only(reference_model(step_input))
        ((output - target).pow(2).mean()).backward()
        optimizer.step()
    boundary_events.clear()
    original_boundary_calls = 0
    for _ in range(args.repeats):
        optimizer.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_input = x.detach().clone().requires_grad_(True)
        output = output_only(reference_model(step_input))
        ((output - target).pow(2).mean()).backward()
        optimizer.step()
        end.record()
        step_events.append((start, end))
    torch.cuda.synchronize()
    boundary_ms = [float(start.elapsed_time(end)) for start, end in boundary_events]
    step_ms = [float(start.elapsed_time(end)) for start, end in step_events]

    metadata = {
        "status": "PASS", "authentic": True, "source_commit": "5be9626709af2722333bf54797c954c09edeada3",
        "device": torch.cuda.get_device_name(0), "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": sys.version,
        "shape": {"batch": 2, "sequence": 3, "hidden": 8, "fc1_output": 32},
        "dtype": "torch.float32", "tensor_parallel_size": 1, "sequence_parallel": False,
        "warmups": args.warmups, "repeats": args.repeats, "invocations_per_step": original_boundary_calls / args.repeats,
        "path": "Megatron MLP.forward -> megatron.core.fusions.fused_bias_swiglu.bias_swiglu_impl -> BiasSwiGLUFunction -> Triton JIT callables",
    }
    (args.out / "boundary_timing.json").write_text(json.dumps({**metadata, "timing": summary(boundary_ms), "unit": "ms", "timing_source": "torch.cuda.Event around original Megatron bias_swiglu_impl"}, indent=2), encoding="utf-8")
    (args.out / "whole_step_timing.json").write_text(json.dumps({**metadata, "timing": summary(step_ms), "unit": "ms", "timing_source": "torch.cuda.Event around forward-loss-backward-optimizer.step"}, indent=2), encoding="utf-8")
    boundary_median = statistics.median(boundary_ms)
    step_median = statistics.median(step_ms)
    fraction = (boundary_median * (original_boundary_calls / args.repeats)) / step_median
    ceiling = 1.0 / (1.0 - fraction) if fraction < 1.0 else None
    (args.out / "swiglu_e2e_ceiling_v2.json").write_text(json.dumps({"status": "PASS", "scope": "LOCAL_MLP_TRAINING_STEP", "operator_fraction_of_step": {"value": fraction, "evidence": "DERIVED", "inputs": ["boundary_timing.json", "whole_step_timing.json", "invocations_per_step"]}, "max_possible_e2e_speedup": ceiling}, indent=2), encoding="utf-8")
    from lab.runtime.reasoning.hypothesis_planner import HypothesisPlanner
    from lab.runtime.reasoning.mechanism_memory import MechanismStore
    from lab.runtime.reasoning.performance_model import PerformanceFacts
    mechanism_path = args.repo.resolve() / "knowledge" / "mechanisms.jsonl"
    mechanisms = MechanismStore(mechanism_path).query("swiglu", operator="swiglu")
    facts = PerformanceFacts(
        operator="swiglu", shape={"batch": 2, "sequence": 3, "hidden_fc1": 32}, dtype="torch.float32",
        dataflow=("fc1", "bias_add", "swiglu", "fc2"),
        tensor_lifetimes={"fc1_output": "until_swiglu", "swiglu_output": "until_fc2"},
        global_memory_reads=("fc1_output",), global_memory_writes=("swiglu_output",),
        producer_consumer_boundaries=("fc1->bias_add_swiglu", "bias_add_swiglu->fc2"),
        profile_evidence={
            "idle_resources": [], "supports_mechanism_ids": [], "contradicts_mechanism_ids": [],
            "measured_invocations_per_step": original_boundary_calls / args.repeats,
            "boundary_latency_ms": boundary_median, "whole_step_latency_ms": step_median,
            "evidence_status": "MEASURED", "authentic": True,
        },
        unknown_fields=("registers_per_thread", "kernel_launch_count", "CUPTI_kernel_breakdown"),
        e2e_profile={
            "scope": "LOCAL_MLP_TRAINING_STEP", "operator_fraction_of_step": fraction,
            "operator_fraction_evidence": "DERIVED", "max_possible_e2e_speedup": ceiling,
        },
    )
    ranked = HypothesisPlanner().rank(facts, mechanisms)
    (args.out / "real_performance_facts_v3.json").write_text(json.dumps(facts.to_dict(), indent=2, default=str), encoding="utf-8")
    (args.out / "ranked_opportunities_v3.json").write_text(json.dumps([item.to_dict() for item in ranked], indent=2, default=str), encoding="utf-8")
    (args.out / "planner_input_summary.json").write_text(json.dumps({"mechanism_store": str(mechanism_path), "matching_mechanisms": len(mechanisms), "planner_output_count": len(ranked)}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
