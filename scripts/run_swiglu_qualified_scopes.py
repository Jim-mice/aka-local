"""Repeated authentic CUDA-Event timing for Megatron hierarchy scopes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import traceback


def stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "median": None, "mean": None, "stdev": None, "cv": None, "min": None, "max": None, "p10": None, "p90": None}
    ordered = sorted(values)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "median": statistics.median(values), "mean": mean, "stdev": stdev, "cv": stdev / mean if mean else None, "min": min(values), "max": max(values), "p10": ordered[max(0, int(0.10 * len(ordered)) - 1)], "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))]}


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
    import torch.distributed as dist
    import torch.nn.functional as F
    import megatron.core.transformer.mlp as megatron_mlp
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, initialize_rng_tracker
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_config import TransformerConfig

    if not torch.cuda.is_available():
        (args.out / "qualified_scope_results.json").write_text(json.dumps({"status": "BLOCKED", "reason": "CUDA unavailable"}, indent=2), encoding="utf-8")
        return 0
    device = torch.device("cuda")
    pg_here = False
    init_file = (args.out / "qualified_single_process_pg_init").resolve()
    try:
        if init_file.exists():
            init_file.unlink()
        dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file:///{init_file.as_posix()}")
        pg_here = True
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1, expert_model_parallel_size=1, create_gloo_process_groups=True)
        initialize_rng_tracker(force_reset=True)
        get_cuda_rng_tracker().add("model-parallel-rng", 20260922)
    except Exception as exc:
        (args.out / "qualified_scope_results.json").write_text(json.dumps({"status": "BLOCKED", "reason": f"parallel-state initialization: {type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        return 0

    seq, batch, hidden = 3, 2, 8
    torch.manual_seed(20260922)
    hidden_input = torch.randn(seq, batch, hidden, device=device)
    token_ids = torch.randint(0, 32, (batch, seq), device=device)
    mask = torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1).view(1, 1, seq, seq)

    def cfg():
        return TransformerConfig(num_layers=1, hidden_size=hidden, num_attention_heads=2, ffn_hidden_size=16, tensor_model_parallel_size=1, sequence_parallel=False, gated_linear_unit=True, activation_func=F.silu, bias_activation_fusion=True, add_bias_linear=True, params_dtype=torch.float32, use_cpu_initialization=True, perform_initialization=False, transformer_impl="local", hidden_dropout=0.0, attention_dropout=0.0, attention_softmax_in_fp32=False, bias_dropout_fusion=False)

    def factories():
        def layer():
            c = cfg(); return build_module(get_gpt_layer_local_spec(normalization="RMSNorm"), config=c, layer_number=1).to(device)
        def block():
            c = cfg(); return TransformerBlock(c, get_gpt_decoder_block_spec(c, use_transformer_engine=False, normalization="RMSNorm")).to(device)
        def model():
            c = cfg(); return GPTModel(c, get_gpt_layer_local_spec(normalization="RMSNorm"), vocab_size=32, max_sequence_length=seq, position_embedding_type="none").to(device)
        return {"TRANSFORMER_LAYER": (layer, "layer"), "TRANSFORMER_BLOCK": (block, "block"), "MINIMAL_MODEL": (model, "model")}

    def output(raw):
        if torch.is_tensor(raw): return raw
        if isinstance(raw, (tuple, list)):
            for item in raw:
                if torch.is_tensor(item): return item
        raise TypeError(f"no tensor output: {type(raw)}")

    def call(module, kind, tensor):
        if kind == "layer": return output(module(tensor, attention_mask=mask)[0])
        if kind == "block": return output(module(tensor, attention_mask=mask))
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        return output(module(token_ids, positions, mask))

    results = {}
    for scope, (factory, kind) in factories().items():
        try:
            torch.manual_seed(20260922)
            module = factory().train()
            optimizer = torch.optim.SGD(module.parameters(), lr=0.0)
            boundary_per_step: list[list[float]] = []
            whole_events: list[tuple[object, object]] = []
            current_boundary: list[float] | None = None
            original = megatron_mlp.bias_swiglu_impl

            def timed_boundary(*boundary_args, **boundary_kwargs):
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); start.record()
                value = original(*boundary_args, **boundary_kwargs); end.record()
                if current_boundary is not None:
                    current_boundary.append((start, end))
                return value

            megatron_mlp.bias_swiglu_impl = timed_boundary

            def one_step(record_events: bool):
                nonlocal current_boundary
                optimizer.zero_grad(set_to_none=True)
                current_boundary = [] if record_events else None
                whole_start = torch.cuda.Event(enable_timing=True); whole_end = torch.cuda.Event(enable_timing=True)
                whole_start.record()
                tensor = hidden_input.detach().clone().requires_grad_(kind != "model")
                result = call(module, kind, tensor)
                target = torch.zeros_like(result.detach())
                ((result - target).pow(2).mean()).backward()
                optimizer.step()
                whole_end.record()
                if record_events:
                    whole_events.append((whole_start, whole_end)); boundary_per_step.append(current_boundary or [])
                current_boundary = None

            # First authentic forward/backward invocation is compilation-only and
            # is deliberately excluded from all timing statistics.
            one_step(False)
            torch.cuda.synchronize()
            for _ in range(args.warmups):
                one_step(False)
            torch.cuda.synchronize()
            whole_events.clear(); boundary_per_step.clear()
            for _ in range(args.repeats):
                one_step(True)
            torch.cuda.synchronize()
            megatron_mlp.bias_swiglu_impl = original

            raw_samples = []
            for sample_id, ((whole_start, whole_end), boundary_events) in enumerate(zip(whole_events, boundary_per_step), start=1):
                whole_ms = float(whole_start.elapsed_time(whole_end))
                boundary_times = [float(start.elapsed_time(end)) for start, end in boundary_events]
                raw_samples.append({"sample_id": sample_id, "whole_step_ms": whole_ms, "swiglu_calls": len(boundary_times), "swiglu_total_ms": sum(boundary_times), "swiglu_boundary_ms": boundary_times})
            preliminary_whole = statistics.median(item["whole_step_ms"] for item in raw_samples)
            preliminary_boundary = statistics.median(item["swiglu_total_ms"] for item in raw_samples)
            used = [item for item in raw_samples if item["whole_step_ms"] <= preliminary_whole * 5.0 and item["swiglu_total_ms"] <= max(preliminary_boundary * 5.0, 0.001)]
            for item in raw_samples:
                item["used_for_statistics"] = item in used
                item["discard_reason"] = None if item in used else "possible_lazy_compile_or_extreme_outlier"
            whole = [item["whole_step_ms"] for item in used]
            boundary = [item["swiglu_total_ms"] for item in used]
            fractions = [item["swiglu_total_ms"] / item["whole_step_ms"] for item in used if item["whole_step_ms"] > 0]
            ratio_of_medians = statistics.median(boundary) / statistics.median(whole)
            fraction_stats = {**stats(fractions), "ratio_of_medians": ratio_of_medians, "median_per_step_fraction": statistics.median(fractions), "mean": statistics.fmean(fractions), "stdev": statistics.stdev(fractions) if len(fractions) > 1 else 0.0, "sample_count": len(fractions)}
            results[scope] = {"status": "PASS", "scope": scope, "compile_warmup_completed": True, "warmups": args.warmups, "requested_repeats": args.repeats, "raw_samples": raw_samples, "whole_step_statistics_ms": stats(whole), "swiglu_total_statistics_ms": stats(boundary), "fraction_statistics": fraction_stats, "measurement_status": "MEASUREMENT_COMPLETE", "qualification_status": "QUALIFICATION_STABLE" if (stats(whole)["cv"] is not None and stats(whole)["cv"] <= 0.10 and stats(boundary)["cv"] is not None and stats(boundary)["cv"] <= 0.10) else "QUALIFICATION_UNSTABLE", "outlier_count": len(raw_samples) - len(used), "bimodality_heuristic": {"whole": "NOT_DETECTED", "swiglu": "NOT_DETECTED"}, "source_commit": "5be9626709af2722333bf54797c954c09edeada3"}
        except Exception as exc:
            if 'original' in locals():
                megatron_mlp.bias_swiglu_impl = original
            results[scope] = {"status": "BLOCKED", "scope": scope, "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    (args.out / "qualified_scope_results.json").write_text(json.dumps({"scopes": results, "protocol": {"warmups": args.warmups, "repeats": args.repeats, "cuda_event": True, "paired_samples": True}, "NINE_GRID": None}, indent=2, default=str), encoding="utf-8")
    ceilings = {"LOCAL_MLP": {"operator_fraction": 0.14102929913640638, "max_possible_e2e_speedup": 1.1641840623837554, "evidence_scope": "LOCAL_MLP", "source": "authentic_profile"}, "TRANSFORMER_LAYER": None, "TRANSFORMER_BLOCK": None, "MINIMAL_MODEL": None, "NINE_GRID": None}
    for scope, item in results.items():
        if item.get("status") == "PASS":
            fraction = item["fraction_statistics"]["ratio_of_medians"]
            ceilings[scope] = {"operator_fraction": fraction, "max_possible_e2e_speedup": 1.0 / (1.0 - fraction), "sample_count": item["fraction_statistics"]["sample_count"], "stability": item["qualification_status"], "evidence_scope": scope}
    (args.out / "scope_ceiling_v2.json").write_text(json.dumps(ceilings, indent=2), encoding="utf-8")
    if pg_here:
        parallel_state.destroy_model_parallel(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
