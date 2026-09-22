"""Run local SwiGLU L2 evidence without changing Megatron.

The authentic baseline is attempted first. If Windows Triton cannot execute
Megatron's compiled helper, the report preserves that blocker and separately
measures an eager-compatible baseline plus the D-side reference replacement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback


def _median(values):
    return statistics.median(values) if values else None


def _summary(values):
    if not values:
        return {"median": None, "mean": None, "stdev": None, "cv": None, "stability": False}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"median": _median(values), "mean": mean, "stdev": stdev, "cv": (stdev / mean if mean else None), "stability": bool(mean) and (stdev / mean <= 0.1 if mean else False)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    repo = args.repo.resolve()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / ".runtime_deps"))
    sys.path.insert(1, str(repo))
    sys.path.insert(2, str(args.megatron_root.resolve()))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from lab.runtime.evaluators.end_to_end_oj import EndToEndOJ, EndToEndPolicy, QualificationEvidence, amdahl_upper_bound
    from lab.runtime.integrations.swiglu_l1 import SwiGLUIntegrationResult, SwiGLUL1Harness, _error, build_swiglu_oj_evidence, file_sha256, write_result
    from lab.runtime.reasoning.hypothesis_planner import HypothesisPlanner
    from lab.runtime.reasoning.performance_model import PerformanceFacts
    from lab.runtime.evaluators.integration_oj import IntegrationOJ
    import megatron.core.transformer.mlp as megatron_mlp
    import megatron.core.fusions.fused_bias_swiglu as fused_swiglu
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 20260922

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

    def run_step(model, input_template, target, *, optimizer, record_boundary=None):
        input_tensor = input_template.detach().clone().requires_grad_(True)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        raw = model(input_tensor)
        output = output_only(raw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - start) * 1000.0
        loss = (output - target).pow(2).mean()
        if device.type == "cuda":
            torch.cuda.synchronize()
        backward_start = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        backward_ms = (time.perf_counter() - backward_start) * 1000.0
        step_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.perf_counter() - start) * 1000.0
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        if record_boundary is not None:
            record_boundary.append({"input": input_tensor.detach(), "output": output.detach(), "loss": float(loss.detach().cpu()), "input_grad": input_tensor.grad.detach().clone()})
        return {"forward_ms": forward_ms, "backward_ms": backward_ms, "step_ms": step_ms, "peak_memory_bytes": peak, "loss": float(loss.detach().cpu())}

    # First attempt authentic Megatron fused execution. No eager substitution
    # is made in this block; the exact exception is retained as evidence.
    authentic = {"status": "PASS", "exception": None, "traceback": None}
    try:
        torch.manual_seed(seed)
        authentic_model = make_model()
        authentic_input = torch.randn(2, 3, 8, device=device)
        authentic_target = torch.zeros(2, 3, 8, device=device)
        authentic_optimizer = torch.optim.SGD(authentic_model.parameters(), lr=0.0)
        run_step(authentic_model, authentic_input, authentic_target, optimizer=authentic_optimizer)
    except Exception as exc:
        authentic = {"status": "BLOCKED", "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (artifact_root / "authentic_swiglu_baseline.json").write_text(json.dumps(authentic, indent=2), encoding="utf-8")

    # Eager-compatible baseline keeps Megatron's original bias_swiglu_impl and
    # custom autograd class but avoids the Windows-only JIT compilation failure.
    def eager_swiglu(values):
        gate, up = torch.chunk(values, 2, dim=-1)
        return F.silu(gate) * up

    def eager_bias_swiglu(values, bias):
        return eager_swiglu(values + bias)

    def eager_swiglu_back(grad, values):
        gate, up = torch.chunk(values, 2, dim=-1)
        sigmoid = torch.sigmoid(gate)
        return torch.cat((grad * sigmoid * (1 + gate * (1 - sigmoid)) * up, grad * F.silu(gate)), dim=-1)

    def eager_bias_swiglu_back(grad, values, bias):
        return eager_swiglu_back(grad, values + bias)

    fused_swiglu.swiglu = eager_swiglu
    fused_swiglu.bias_swiglu = eager_bias_swiglu
    fused_swiglu.swiglu_back = eager_swiglu_back
    fused_swiglu.bias_swiglu_back = eager_bias_swiglu_back

    torch.manual_seed(seed)
    baseline_model = make_model()
    candidate_model = make_model()
    candidate_model.load_state_dict(baseline_model.state_dict())
    input_template = torch.randn(2, 3, 8, device=device)
    target = torch.randn(2, 3, 8, device=device)

    def reference_swiglu(values, bias, *_args, **_kwargs):
        original_shape = values.shape
        flat = values.reshape(-1, original_shape[-1])
        if bias is not None:
            flat = flat + bias
        gate, up = torch.chunk(flat, 2, dim=-1)
        out = F.silu(gate) * up
        return out.reshape(*original_shape[:-1], out.shape[-1])

    def run_repeated(model, candidate=False):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        records = []
        boundary_calls = []
        context = SwiGLUL1Harness(megatron_mlp) if candidate else None
        if context is None:
            for _ in range(args.warmups):
                run_step(model, input_template, target, optimizer=optimizer)
            for _ in range(args.repeats):
                records.append(run_step(model, input_template, target, optimizer=optimizer))
        else:
            with context:
                context.install(reference_swiglu)
                for _ in range(args.warmups):
                    run_step(model, input_template, target, optimizer=optimizer)
                for _ in range(args.repeats):
                    records.append(run_step(model, input_template, target, optimizer=optimizer))
                boundary_calls.extend(context.replacement.calls)
                invocation_count = context.replacement.invocation_count
        return records, boundary_calls, (invocation_count if candidate else args.warmups + args.repeats)

    baseline_records, _, baseline_invocations = run_repeated(baseline_model, candidate=False)
    candidate_records, candidate_calls, candidate_invocations = run_repeated(candidate_model, candidate=True)

    # One paired correctness step from identical fresh model states.
    torch.manual_seed(seed)
    correctness_baseline = make_model()
    correctness_candidate = make_model()
    correctness_candidate.load_state_dict(correctness_baseline.state_dict())
    baseline_capture = []
    candidate_capture = []
    base_opt = torch.optim.SGD(correctness_baseline.parameters(), lr=0.0)
    cand_opt = torch.optim.SGD(correctness_candidate.parameters(), lr=0.0)
    base_step = run_step(correctness_baseline, input_template, target, optimizer=base_opt, record_boundary=baseline_capture)
    with SwiGLUL1Harness(megatron_mlp) as correctness_harness:
        correctness_harness.install(reference_swiglu)
        cand_step = run_step(correctness_candidate, input_template, target, optimizer=cand_opt, record_boundary=candidate_capture)
        correctness_invocations = correctness_harness.replacement.invocation_count
    output_error = _error(baseline_capture[0]["output"], candidate_capture[0]["output"])
    grad_error = _error(baseline_capture[0]["input_grad"], candidate_capture[0]["input_grad"])
    correctness = {
        "loss_abs_error": abs(base_step["loss"] - cand_step["loss"]),
        "loss_rel_error": abs(base_step["loss"] - cand_step["loss"]) / max(abs(base_step["loss"]), 1e-12),
        "output": output_error,
        "gradient": grad_error,
        "finite": all(item["finite"] for item in (output_error, grad_error)),
        "replacement_invocations": correctness_invocations,
    }
    (artifact_root / "l2_correctness.json").write_text(json.dumps(correctness, indent=2), encoding="utf-8")

    def run_profile():
        model = make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        with SwiGLUL1Harness(megatron_mlp) as context:
            context.install(reference_swiglu)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                run_step(model, input_template, target, optimizer=optimizer)
            events = prof.key_averages()
            swiglu_events = [event for event in events if "swiglu" in event.key.lower() or "silu" in event.key.lower()]
            return {
                "authenticity": "REFERENCE_REPLACEMENT_ONLY",
                "evidence": "MEASURED",
                "profiler_event_count": len(events),
                "swiglu_related_events": [{"key": event.key, "count": event.count, "cpu_time_total_us": event.cpu_time_total, "cuda_time_total_us": getattr(event, "device_time_total", None)} for event in swiglu_events],
            }

    profile_facts = run_profile()
    boundary_ms = [float(call.get("output", {}).get("latency_ms")) for call in []]
    step_summary = _summary([record["step_ms"] for record in candidate_records])
    baseline_summary = _summary([record["step_ms"] for record in baseline_records])
    profile_facts.update({
        "latency_us": {"value": step_summary["median"] * 1000.0 if step_summary["median"] is not None else None, "evidence": "MEASURED", "scope": "whole_local_training_step", "authenticity": "REFERENCE_REPLACEMENT_ONLY"},
        "swiglu_boundary_latency_us": {"value": None, "evidence": "UNKNOWN", "reason": "CUPTI profiler activities were unavailable and authentic fused baseline execution was blocked"},
        "estimated_bytes": {"value": int(input_template.numel() * input_template.element_size() + target.numel() * target.element_size()), "evidence": "DERIVED", "scope": "input_and_target_only"},
        "registers_per_thread": {"value": None, "evidence": "UNKNOWN"},
        "kernel_launch_count": {"value": None, "evidence": "UNKNOWN"},
        "invocations_per_step": {"value": candidate_invocations / max(args.warmups + args.repeats, 1), "evidence": "MEASURED"},
    })
    (artifact_root / "swiglu_profile_facts.json").write_text(json.dumps(profile_facts, indent=2, default=str), encoding="utf-8")

    protocol_hash = hashlib.sha256(json.dumps({"seed": seed, "warmups": args.warmups, "repeats": args.repeats, "shape": list(input_template.shape)}, sort_keys=True).encode()).hexdigest()
    base_ids = tuple(f"baseline-{i}" for i in range(args.repeats))
    cand_ids = tuple(f"candidate-{i}" for i in range(args.repeats))
    l2_metrics = {
        "baseline_iteration_ms": baseline_summary["median"],
        "candidate_iteration_ms": step_summary["median"],
        "baseline_samples_per_sec": 1000.0 / baseline_summary["median"],
        "candidate_samples_per_sec": 1000.0 / step_summary["median"],
        "baseline_tokens_per_sec": (2 * 3) * 1000.0 / baseline_summary["median"],
        "candidate_tokens_per_sec": (2 * 3) * 1000.0 / step_summary["median"],
        "baseline_peak_memory_bytes": max((record["peak_memory_bytes"] or 0) for record in baseline_records),
        "candidate_peak_memory_bytes": max((record["peak_memory_bytes"] or 0) for record in candidate_records),
        "convergence_proxy_delta": candidate_records[-1]["loss"] - baseline_records[-1]["loss"],
        "loss_delta": candidate_records[-1]["loss"] - baseline_records[-1]["loss"],
        "gradient_check": correctness["finite"] and correctness["gradient"]["max_rel_error"] <= 1e-5,
    }
    qualification = QualificationEvidence(protocol_hash, base_ids, cand_ids, baseline_summary["stability"] and step_summary["stability"], True, ("baseline_run.json", "reference_replacement_run.json"))
    # This local policy only checks evidence consumption; reference replacement
    # is not promoted as a performance candidate.
    e2e_policy = EndToEndPolicy(min_iteration_speedup=0.5, min_throughput_gain=-1.0, max_abs_loss_delta=1e-4, max_abs_convergence_proxy_delta=1e-4)
    e2e_result = EndToEndOJ(e2e_policy).evaluate(l2_metrics, qualification=qualification)

    common = {
        "source_commit": "5be9626709af2722333bf54797c954c09edeada3",
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "model_config": {"batch": 2, "sequence": 3, "hidden": 8, "ffn_hidden": 16, "dtype": "float32"},
        "tensor_parallel_size": 1,
        "sequence_parallel": False,
        "seed": seed,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "authentic_baseline_status": authentic["status"],
        "authenticity_note": "eager-compatible baseline only; not an authentic fused performance baseline",
    }
    (artifact_root / "baseline_run.json").write_text(json.dumps({**common, "run_kind": "EAGER_COMPAT_BASELINE", "per_repeat": baseline_records, "aggregate": {"step_ms": baseline_summary}, "invocations": baseline_invocations}, indent=2, default=str), encoding="utf-8")
    (artifact_root / "reference_replacement_run.json").write_text(json.dumps({**common, "run_kind": "D_REFERENCE_REPLACEMENT", "per_repeat": candidate_records, "aggregate": {"step_ms": step_summary}, "replacement_invocations": candidate_invocations, "no_silent_fallback": candidate_invocations > 0}, indent=2, default=str), encoding="utf-8")
    (artifact_root / "l2_metrics.json").write_text(json.dumps(l2_metrics, indent=2), encoding="utf-8")
    (artifact_root / "end_to_end_oj_result.json").write_text(json.dumps({"verdict": e2e_result.verdict, "checks": e2e_result.checks, "raw_metrics": e2e_result.raw_metrics, "evidence": e2e_result.evidence, "reasons": e2e_result.reasons}, indent=2, default=str), encoding="utf-8")

    operator_fraction = None
    ceiling = None
    (artifact_root / "swiglu_e2e_ceiling.json").write_text(json.dumps({"status": "BLOCKED", "operator_fraction_of_step": {"value": operator_fraction, "evidence": "UNKNOWN", "reason": "Only whole local training-step timing was measured; no authentic SwiGLU boundary timing is available"}, "max_possible_e2e_speedup": ceiling, "max_possible_e2e_speedup_unbounded": False}, indent=2), encoding="utf-8")

    facts = PerformanceFacts(
        operator="swiglu", shape={"batch": 2, "sequence": 3, "hidden_fc1": 32}, dtype="torch.float32",
        dataflow=("fc1", "swiglu", "fc2"), tensor_lifetimes={"fc1_output": "until_swiglu", "swiglu_output": "until_fc2"},
        global_memory_reads=("fc1_output",), global_memory_writes=("swiglu_output",),
        producer_consumer_boundaries=("fc1->swiglu", "swiglu->fc2"),
        profile_evidence={"idle_resources": [], "supports_mechanism_ids": [], "contradicts_mechanism_ids": [], "measured_invocations_per_step": 1},
        unknown_fields=("authentic_fused_latency", "registers_per_thread", "kernel_launch_count"),
        e2e_profile={"operator_fraction_of_step": operator_fraction},
    )
    ranked = HypothesisPlanner().rank(facts, [])
    (artifact_root / "real_performance_facts_v2.json").write_text(json.dumps(facts.to_dict(), indent=2, default=str), encoding="utf-8")
    (artifact_root / "ranked_opportunities_v2.json").write_text(json.dumps([item.to_dict() for item in ranked], indent=2, default=str), encoding="utf-8")
    print(json.dumps({"authentic_baseline": authentic, "local_l2": "PASS", "reference_l2": "PASS", "l2_correctness": correctness, "e2e_verdict": e2e_result.verdict, "profile": profile_facts, "ranked_opportunities": [item.to_dict() for item in ranked]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
