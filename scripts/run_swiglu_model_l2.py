"""Bring up authentic Megatron Transformer scopes without modifying Megatron."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo.resolve() / ".runtime_deps"))
    sys.path.insert(1, str(args.repo.resolve()))
    sys.path.insert(2, str(args.megatron_root.resolve()))

    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    import megatron.core.transformer.mlp as megatron_mlp
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_config import TransformerConfig
    from lab.runtime.integrations.swiglu_l1 import SwiGLUL1Harness
    from lab.runtime.evaluators.end_to_end_oj import EndToEndOJ, EndToEndPolicy, QualificationEvidence
    import hashlib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 20260922
    base_info = {
        "source_commit": "5be9626709af2722333bf54797c954c09edeada3",
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "device": str(device),
        "call_chain": "GPTModel.forward -> TransformerBlock.forward -> TransformerLayer.forward -> TransformerLayer._forward_mlp -> MLP.forward -> bias_swiglu_impl",
        "source_locations": {
            "gpt_model": "megatron/core/models/gpt/gpt_model.py::GPTModel.forward",
            "transformer_block": "megatron/core/transformer/transformer_block.py::TransformerBlock.forward",
            "transformer_layer": "megatron/core/transformer/transformer_layer.py::TransformerLayer.forward",
            "mlp": "megatron/core/transformer/mlp.py::MLP.forward",
            "swiglu": "megatron/core/fusions/fused_bias_swiglu.py::bias_swiglu_impl",
        },
    }
    (args.out / "real_hierarchy.json").write_text(json.dumps(base_info, indent=2), encoding="utf-8")
    if device.type != "cuda":
        (args.out / "scope_results.json").write_text(json.dumps({"status": "BLOCKED", "reason": "CUDA unavailable", "scopes": {}}, indent=2), encoding="utf-8")
        return 0

    process_group_initialized_here = False
    try:
        from megatron.core import parallel_state
        if not dist.is_initialized():
            init_file = (args.out / "single_process_pg_init").resolve()
            if init_file.exists():
                init_file.unlink()
            dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file:///{init_file.as_posix()}")
            process_group_initialized_here = True
        if not parallel_state.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                context_parallel_size=1, expert_model_parallel_size=1,
                create_gloo_process_groups=True,
            )
        from megatron.core.tensor_parallel.random import initialize_rng_tracker, get_cuda_rng_tracker
        initialize_rng_tracker(force_reset=True)
        get_cuda_rng_tracker().add("model-parallel-rng", 20260922)
    except Exception as exc:
        (args.out / "scope_results.json").write_text(json.dumps({"status": "BLOCKED", "reason": f"parallel-state initialization: {type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "scopes": {}}, indent=2), encoding="utf-8")
        return 0

    def config():
        return TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=2, ffn_hidden_size=16,
            tensor_model_parallel_size=1, sequence_parallel=False, gated_linear_unit=True,
            activation_func=F.silu, bias_activation_fusion=True, add_bias_linear=True,
            params_dtype=torch.float32, use_cpu_initialization=True, perform_initialization=False,
            transformer_impl="local", hidden_dropout=0.0, attention_dropout=0.0,
            attention_softmax_in_fp32=False, bias_dropout_fusion=False,
        )

    def layer_factory():
        cfg = config()
        spec = get_gpt_layer_local_spec(normalization="RMSNorm")
        return build_module(spec, config=cfg, layer_number=1).to(device)

    def block_factory():
        cfg = config()
        spec = get_gpt_decoder_block_spec(cfg, use_transformer_engine=False, normalization="RMSNorm")
        return TransformerBlock(cfg, spec).to(device)

    def model_factory():
        cfg = config()
        spec = get_gpt_layer_local_spec(normalization="RMSNorm")
        return GPTModel(cfg, spec, vocab_size=32, max_sequence_length=3, position_embedding_type="none").to(device)

    seq, batch, hidden = 3, 2, 8
    x = torch.randn(seq, batch, hidden, device=device)
    target = torch.randn(seq, batch, hidden, device=device)
    # True entries are masked by Megatron's local attention implementation.
    mask = torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1).view(1, 1, seq, seq)

    def tensor_output(raw):
        if torch.is_tensor(raw):
            return raw
        if isinstance(raw, (tuple, list)):
            for item in raw:
                if torch.is_tensor(item):
                    return item
        raise TypeError(f"no tensor output in {type(raw)}")

    def invoke(module, kind, input_tensor, ids=None):
        if kind == "layer":
            return tensor_output(module(input_tensor, attention_mask=mask)[0])
        if kind == "block":
            return tensor_output(module(input_tensor, attention_mask=mask))
        position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        return tensor_output(module(ids, position_ids, mask))

    def run_scope(name, factory, kind):
        record = {"status": "PASS", "scope": name, **base_info, "shape": [seq, batch, hidden], "replacement_invocations": 0}
        try:
            torch.manual_seed(seed)
            baseline = factory()
            candidate = factory()
            candidate.load_state_dict(baseline.state_dict())
            baseline_input = x.detach().clone().requires_grad_(True)
            model_ids = torch.randint(0, 32, (batch, seq), device=device)
            baseline_events = []
            original_boundary = megatron_mlp.bias_swiglu_impl

            def timed_boundary(*boundary_args, **boundary_kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                value = original_boundary(*boundary_args, **boundary_kwargs)
                end.record()
                baseline_events.append((start, end))
                return value

            megatron_mlp.bias_swiglu_impl = timed_boundary
            try:
                start = torch.cuda.Event(enable_timing=True); start.record()
                baseline_output = invoke(baseline, kind, baseline_input, model_ids)
                scope_target = torch.zeros_like(baseline_output.detach())
                baseline_loss = (baseline_output - scope_target).pow(2).mean()
                baseline_loss.backward()
                end = torch.cuda.Event(enable_timing=True); end.record()
                torch.cuda.synchronize()
                baseline_ms = float(start.elapsed_time(end))
                boundary_ms = [float(a.elapsed_time(b)) for a, b in baseline_events]
                baseline_grad = (baseline_input.grad.detach().clone() if baseline_input.grad is not None else next(parameter.grad.detach().clone() for parameter in baseline.parameters() if parameter.grad is not None))
            finally:
                megatron_mlp.bias_swiglu_impl = original_boundary

            candidate_input = x.detach().clone().requires_grad_(True)
            with SwiGLUL1Harness(megatron_mlp) as harness:
                harness.install(lambda values, bias, *_args, **_kwargs: F.silu((values + bias).chunk(2, dim=-1)[0]) * (values + bias).chunk(2, dim=-1)[1])
                candidate_start = torch.cuda.Event(enable_timing=True); candidate_start.record()
                candidate_output = invoke(candidate, kind, candidate_input, model_ids)
                candidate_loss = (candidate_output - scope_target).pow(2).mean()
                candidate_loss.backward()
                candidate_end = torch.cuda.Event(enable_timing=True); candidate_end.record()
                torch.cuda.synchronize()
                record["replacement_invocations"] = harness.replacement.invocation_count
            candidate_grad = (candidate_input.grad.detach().clone() if candidate_input.grad is not None else next(parameter.grad.detach().clone() for parameter in candidate.parameters() if parameter.grad is not None))
            candidate_ms = float(candidate_start.elapsed_time(candidate_end))
            record.update({
                "forward_max_abs_error": float((baseline_output - candidate_output).abs().max().cpu()),
                "forward_max_rel_error": float(((baseline_output - candidate_output).abs() / candidate_output.abs().clamp_min(1e-12)).max().cpu()),
                "loss_abs_error": float((baseline_loss - candidate_loss).abs().cpu()),
                "gradient_kind": "input" if candidate_input.grad is not None else "first_parameter",
                "input_grad_max_abs_error": float((baseline_grad - candidate_grad).abs().max().cpu()),
                "finite": bool(torch.isfinite(baseline_output).all() and torch.isfinite(candidate_output).all() and torch.isfinite(baseline_grad).all() and torch.isfinite(candidate_grad).all()),
                "no_silent_fallback": record["replacement_invocations"] > 0,
                "authentic_boundary_ms": boundary_ms,
                "authentic_boundary_median_ms": statistics.median(boundary_ms) if boundary_ms else None,
                "whole_scope_ms": baseline_ms,
                "replacement_whole_scope_ms": candidate_ms,
                "operator_fraction": (statistics.median(boundary_ms) / baseline_ms) if boundary_ms and baseline_ms else None,
            })
            if record["replacement_invocations"] <= 0:
                record["status"] = "FAIL"
        except Exception as exc:
            record = {**record, "status": "BLOCKED", "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        return record

    results = {}
    for name, factory, kind in (("TRANSFORMER_LAYER", layer_factory, "layer"), ("TRANSFORMER_BLOCK", block_factory, "block"), ("MINIMAL_MODEL", model_factory, "model")):
        results[name] = run_scope(name, factory, kind)
    (args.out / "scope_results.json").write_text(json.dumps({"scopes": results, "NINE_GRID_E2E": None}, indent=2, default=str), encoding="utf-8")
    oj_results = {}
    for name, record in results.items():
        if record.get("status") != "PASS":
            oj_results[name] = {"verdict": "SYSTEM_PROVISIONAL", "reasons": ["scope_runtime_blocked"]}
            continue
        baseline_ms = record["whole_scope_ms"]
        candidate_ms = record["replacement_whole_scope_ms"]
        metrics = {
            "baseline_iteration_ms": baseline_ms, "candidate_iteration_ms": candidate_ms,
            "baseline_samples_per_sec": 1000.0 / baseline_ms, "candidate_samples_per_sec": 1000.0 / candidate_ms,
            "baseline_tokens_per_sec": seq * batch * 1000.0 / baseline_ms, "candidate_tokens_per_sec": seq * batch * 1000.0 / candidate_ms,
            "baseline_peak_memory_bytes": 0, "candidate_peak_memory_bytes": 0,
            "convergence_proxy_delta": record["loss_abs_error"], "loss_delta": record["loss_abs_error"],
            "gradient_check": record["finite"] and record["input_grad_max_abs_error"] <= 1e-5,
            "operator_fraction_of_step": record["operator_fraction"],
        }
        protocol_hash = hashlib.sha256(json.dumps({"scope": name, "seed": seed}, sort_keys=True).encode()).hexdigest()
        qualification = QualificationEvidence(protocol_hash, (name + "-baseline",), (name + "-replacement",), True, True, ("scope_results.json",))
        policy = EndToEndPolicy(min_iteration_speedup=0.01, min_throughput_gain=-1.0, max_abs_loss_delta=1e-5, max_abs_convergence_proxy_delta=1e-5, require_repeated_qualification=False, min_qualification_runs=1)
        judged = EndToEndOJ(policy).evaluate(metrics, qualification=qualification)
        oj_results[name] = {"scope": name, "verdict": judged.verdict, "checks": judged.checks, "raw_metrics": judged.raw_metrics, "reasons": judged.reasons, "note": "wiring/correctness evidence; reference replacement is not promoted as a performance candidate"}
    (args.out / "model_level_end_to_end_oj.json").write_text(json.dumps(oj_results, indent=2, default=str), encoding="utf-8")
    ceilings = {"LOCAL_MLP": {"value": 0.14102929913640638, "source": "authentic_profile/swiglu_e2e_ceiling_v2.json"}, "TRANSFORMER_LAYER": None, "TRANSFORMER_BLOCK": None, "MINIMAL_MODEL": None, "NINE_GRID": None}
    for name, record in results.items():
        if record.get("status") == "PASS" and record.get("operator_fraction") is not None:
            ceilings[name] = {"operator_fraction": record["operator_fraction"], "max_possible_e2e_speedup": 1.0 / (1.0 - record["operator_fraction"]), "scope": name}
    (args.out / "scope_ceiling.json").write_text(json.dumps(ceilings, indent=2), encoding="utf-8")
    if process_group_initialized_here:
        from megatron.core import parallel_state
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
