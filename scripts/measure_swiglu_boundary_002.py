"""Read-only authentic Megatron SwiGLU boundary measurement harness."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from pathlib import Path


def stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "median": None, "mean": None, "stdev": None, "cv": None, "min": None, "max": None, "p10": None, "p90": None}
    ordered = sorted(values)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "median": statistics.median(values), "mean": mean, "stdev": stdev, "cv": stdev / mean if mean else None, "min": min(values), "max": max(values), "p10": ordered[max(0, int(len(values) * 0.10) - 1)], "p90": ordered[min(len(values) - 1, int(len(values) * 0.90))]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--profile-iteration", action="store_true")
    parser.add_argument("--uninstrumented", action="store_true")
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
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, initialize_rng_tracker
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    environment = {
        "python": sys.executable,
        "python_version": sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_index": 0,
        "source_commit": "5be9626709af2722333bf54797c954c09edeada3",
        "tp": 1,
        "sp": False,
        "batch": 2,
        "sequence": 3,
        "hidden": 8,
        "ffn_hidden": 16,
        "dtype": "float32",
        "warmups": args.warmups,
        "repeats": args.repeats,
        "authentic_path": "Megatron MLP.forward -> bias_swiglu_impl -> BiasSwiGLUFunction -> linear_fc2",
        "source_files": [
            "megatron/core/transformer/mlp.py",
            "megatron/core/fusions/fused_bias_swiglu.py",
        ],
    }
    (args.out / "measurement_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    if not torch.cuda.is_available():
        (args.out / "boundary_timing.json").write_text(json.dumps({"status": "BLOCKED", "reason": "CUDA unavailable", "environment": environment}, indent=2), encoding="utf-8")
        return 0

    pg_here = False
    init_file = (args.out / "measurement_pg_init").resolve()
    try:
        if init_file.exists():
            init_file.unlink()
        dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file:///{init_file.as_posix()}")
        pg_here = True
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1, expert_model_parallel_size=1, create_gloo_process_groups=True)
        initialize_rng_tracker(force_reset=True)
        get_cuda_rng_tracker().add("model-parallel-rng", 20260922)
    except Exception as exc:
        (args.out / "boundary_timing.json").write_text(json.dumps({"status": "BLOCKED", "environment": environment, "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        return 0

    original_boundary = megatron_mlp.bias_swiglu_impl
    try:
        cfg = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2, ffn_hidden_size=16, tensor_model_parallel_size=1, sequence_parallel=False, gated_linear_unit=True, activation_func=F.silu, bias_activation_fusion=True, add_bias_linear=True, params_dtype=torch.float32, use_cpu_initialization=True, perform_initialization=False, transformer_impl="local", hidden_dropout=0.0, attention_dropout=0.0, attention_softmax_in_fp32=False, bias_dropout_fusion=False)
        torch.manual_seed(20260922)
        layer = build_module(get_gpt_layer_local_spec(normalization="RMSNorm"), config=cfg, layer_number=1).to("cuda").eval()
        mlp = layer.mlp
        x = torch.randn(3, 2, 8, device="cuda", dtype=torch.float32)
        nvtx = torch.cuda.nvtx
        ranges_enabled = bool(args.profile_iteration)

        def call_once(record_events: bool = False) -> dict[str, object]:
            if args.uninstrumented:
                whole_start = torch.cuda.Event(enable_timing=True)
                whole_end = torch.cuda.Event(enable_timing=True)
                whole_start.record()
                with torch.no_grad():
                    output = mlp(x)
                whole_end.record()
                return {"whole": (whole_start, whole_end), "events": {}, "output_shape": list(output[0].shape if isinstance(output, tuple) else output.shape)}
            events: dict[str, tuple[object, object]] = {}
            whole_start = torch.cuda.Event(enable_timing=True)
            whole_end = torch.cuda.Event(enable_timing=True)
            active = {"fc1": None, "swiglu": None, "fc2": None}

            def wrapped_fc1(*a, **kw):
                if ranges_enabled: nvtx.range_push("AKA_FC1")
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                if record_events: start.record()
                result = original_fc1(*a, **kw)
                if record_events: end.record(); events["fc1"] = (start, end)
                if ranges_enabled: nvtx.range_pop()
                return result

            def wrapped_fc2(*a, **kw):
                if ranges_enabled: nvtx.range_push("AKA_FC2")
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                if record_events: start.record()
                result = original_fc2(*a, **kw)
                if record_events: end.record(); events["fc2"] = (start, end)
                if ranges_enabled: nvtx.range_pop()
                return result

            def wrapped_swiglu(*a, **kw):
                if ranges_enabled: nvtx.range_push("AKA_SWIGLU")
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                if record_events: start.record()
                result = original_boundary(*a, **kw)
                if record_events: end.record(); events["swiglu"] = (start, end)
                if ranges_enabled: nvtx.range_pop()
                return result

            original_fc1 = mlp.linear_fc1.forward
            original_fc2 = mlp.linear_fc2.forward
            mlp.linear_fc1.forward = wrapped_fc1
            mlp.linear_fc2.forward = wrapped_fc2
            megatron_mlp.bias_swiglu_impl = wrapped_swiglu
            try:
                whole_start.record()
                with torch.no_grad():
                    output = mlp(x)
                whole_end.record()
            finally:
                mlp.linear_fc1.forward = original_fc1
                mlp.linear_fc2.forward = original_fc2
                megatron_mlp.bias_swiglu_impl = original_boundary
            return {"whole": (whole_start, whole_end), "events": events, "output_shape": list(output[0].shape if isinstance(output, tuple) else output.shape)}

        # The first forward compiles the authentic path and is excluded.
        call_once(False)
        torch.cuda.synchronize()
        for _ in range(args.warmups):
            call_once(False)
        torch.cuda.synchronize()
        samples = []
        for i in range(args.repeats):
            samples.append(call_once(True))
        torch.cuda.synchronize()
        raw = []
        for i, sample in enumerate(samples, 1):
            whole_ms = float(sample["whole"][0].elapsed_time(sample["whole"][1]))
            regions = {}
            for name, pair in sample["events"].items():
                regions[name + "_ms"] = float(pair[0].elapsed_time(pair[1]))
            raw.append({"sample_id": i, "whole_mlp_ms": whole_ms, **regions, "output_shape": sample["output_shape"]})
        result = {
            "status": "PASS",
            "environment": environment,
            "compile_warmup_completed": True,
            "event_instrumentation": "PERTURBING",
            "measurement_kind": "authentic Megatron MLP forward; no algorithm replacement",
            "raw_samples": raw,
            "whole_mlp_statistics_ms": stats([x["whole_mlp_ms"] for x in raw]),
            "fc1_statistics_ms": stats([x["fc1_ms"] for x in raw if "fc1_ms" in x]),
            "swiglu_statistics_ms": stats([x["swiglu_ms"] for x in raw if "swiglu_ms" in x]),
            "fc2_statistics_ms": stats([x["fc2_ms"] for x in raw if "fc2_ms" in x]),
            "swiglu_plus_fc2_statistics_ms": stats([x["swiglu_ms"] + x["fc2_ms"] for x in raw if "swiglu_ms" in x and "fc2_ms" in x]),
        }
        (args.out / "boundary_timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        (args.out / "boundary_timing.json").write_text(json.dumps({"status": "BLOCKED", "environment": environment, "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
    finally:
        megatron_mlp.bias_swiglu_impl = original_boundary
        if pg_here:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
