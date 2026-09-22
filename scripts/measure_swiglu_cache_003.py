"""Measurement-only cache and timing attribution for authentic Megatron SwiGLU."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path


def stats(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    mean = statistics.fmean(values) if values else None
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
    return {"count": len(values), "median": statistics.median(values) if values else None, "mean": mean, "stdev": stdev, "cv": (stdev / mean) if mean else None, "min": min(values) if values else None, "max": max(values) if values else None, "p10": ordered[max(0, int(len(ordered) * 0.10) - 1)] if ordered else None, "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))] if ordered else None}


def setup(repo: Path, megatron_root: Path):
    sys.path.insert(0, str(repo.resolve() / ".runtime_deps"))
    sys.path.insert(1, str(repo.resolve()))
    sys.path.insert(2, str(megatron_root.resolve()))
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import megatron.core.transformer.mlp as megatron_mlp
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, initialize_rng_tracker
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_config import TransformerConfig

    out = {"torch": torch, "dist": dist, "F": F, "megatron_mlp": megatron_mlp, "parallel_state": parallel_state, "get_gpt_layer_local_spec": get_gpt_layer_local_spec, "get_cuda_rng_tracker": get_cuda_rng_tracker, "initialize_rng_tracker": initialize_rng_tracker, "build_module": build_module, "TransformerConfig": TransformerConfig}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["full", "ncu-baseline", "ncu-pressure"], default="full")
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measured-batches", type=int, default=50)
    parser.add_argument("--k", type=int, default=8192)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = setup(args.repo, args.megatron_root)
    torch = m["torch"]
    dist = m["dist"]
    F = m["F"]
    megatron_mlp = m["megatron_mlp"]
    parallel_state = m["parallel_state"]

    if not torch.cuda.is_available():
        (args.out / "loop_003_runtime.json").write_text(json.dumps({"status": "BLOCKED", "reason": "CUDA unavailable"}, indent=2), encoding="utf-8")
        return 0
    pg_here = False
    init_file = (args.out / "loop_003_pg_init").resolve()
    try:
        if init_file.exists(): init_file.unlink()
        dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file:///{init_file.as_posix()}")
        pg_here = True
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1, expert_model_parallel_size=1, create_gloo_process_groups=True)
        m["initialize_rng_tracker"](force_reset=True)
        m["get_cuda_rng_tracker"]().add("model-parallel-rng", 20260922)

        cfg = m["TransformerConfig"](num_layers=1, hidden_size=8, num_attention_heads=2, ffn_hidden_size=16, tensor_model_parallel_size=1, sequence_parallel=False, gated_linear_unit=True, activation_func=F.silu, bias_activation_fusion=True, add_bias_linear=True, params_dtype=torch.float32, use_cpu_initialization=True, perform_initialization=False, transformer_impl="local", hidden_dropout=0.0, attention_dropout=0.0, attention_softmax_in_fp32=False, bias_dropout_fusion=False)
        torch.manual_seed(20260922)
        layer = m["build_module"](m["get_gpt_layer_local_spec"](normalization="RMSNorm"), config=cfg, layer_number=1).to("cuda").eval()
        mlp = layer.mlp
        x = torch.randn(3, 2, 8, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            fc1_result = mlp.linear_fc1(x)
            fc1_value, bias = fc1_result[0], fc1_result[1]
            activation = megatron_mlp.bias_swiglu_impl(fc1_value, bias)

        props = torch.cuda.get_device_properties(0)
        l2_value = getattr(props, "L2_cache_size", None)
        environment = {
            "status": "PASS", "python": sys.executable, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "source_commit": "5be9626709af2722333bf54797c954c09edeada3", "batch": 2, "sequence": 3, "hidden": 8, "ffn_hidden": 16, "dtype": str(activation.dtype), "tp": 1, "sp": False, "mode": args.mode, "l2_cache_size_bytes": l2_value,
        }
        footprint = {
            "fc1_value": {"shape": list(fc1_value.shape), "numel": fc1_value.numel(), "dtype": str(fc1_value.dtype), "element_size_bytes": fc1_value.element_size(), "payload_bytes": fc1_value.numel() * fc1_value.element_size(), "stride": list(fc1_value.stride()), "storage_bytes": fc1_value.untyped_storage().nbytes()},
            "bias": {"shape": list(bias.shape), "numel": bias.numel(), "dtype": str(bias.dtype), "payload_bytes": bias.numel() * bias.element_size(), "stride": list(bias.stride()), "storage_bytes": bias.untyped_storage().nbytes()},
            "activation": {"shape": list(activation.shape), "numel": activation.numel(), "dtype": str(activation.dtype), "element_size_bytes": activation.element_size(), "payload_bytes": activation.numel() * activation.element_size(), "stride": list(activation.stride()), "is_contiguous": bool(activation.is_contiguous()), "storage_bytes": activation.untyped_storage().nbytes(), "requires_grad": bool(activation.requires_grad)},
        }
        (args.out / "activation_footprint.json").write_text(json.dumps({"environment": environment, **footprint}, indent=2), encoding="utf-8")

        if args.mode != "full":
            pressure_buffer = None
            pressure_bytes = None
            if args.mode == "ncu-pressure":
                if not l2_value:
                    raise RuntimeError("L2 cache size unavailable; pressure experiment cannot claim an L2-exceeding buffer")
                pressure_bytes = int(l2_value) * 2
                pressure_buffer = torch.empty(pressure_bytes // 4, device="cuda", dtype=torch.float32)
                pressure_buffer.fill_(1.0)
            nvtx = torch.cuda.nvtx
            torch.cuda.synchronize()
            if pressure_buffer is not None:
                pressure_buffer[0] = pressure_buffer.sum()
                torch.cuda.synchronize()
            nvtx.range_push("AKA_FC2_PRESSURE" if args.mode == "ncu-pressure" else "AKA_FC2_BASELINE")
            with torch.no_grad():
                fc2_result = mlp.linear_fc2(activation)
            nvtx.range_pop()
            torch.cuda.synchronize()
            (args.out / (args.mode + "_runtime.json")).write_text(json.dumps({"environment": environment, "pressure_buffer_bytes": pressure_bytes, "fc2_output_shape": list(fc2_result[0].shape), "status": "PASS"}, indent=2), encoding="utf-8")
            return 0

        original = megatron_mlp.bias_swiglu_impl
        event_samples: list[float] = []
        wall_samples: list[float] = []
        for _ in range(20):
            original(fc1_value, bias)
        torch.cuda.synchronize()
        for _ in range(120):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); original(fc1_value, bias); end.record(); torch.cuda.synchronize()
            event_samples.append(float(start.elapsed_time(end)) * 1000.0)
            torch.cuda.synchronize(); begin = time.perf_counter(); original(fc1_value, bias); torch.cuda.synchronize(); wall_samples.append((time.perf_counter() - begin) * 1e6)

        def batched_once() -> float:
            torch.cuda.synchronize(); start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); start.record()
            for _ in range(args.k): original(fc1_value, bias)
            end.record(); torch.cuda.synchronize(); return float(start.elapsed_time(end)) * 1000.0 / args.k

        for _ in range(args.warmup_batches): batched_once()
        batch_samples = [batched_once() for _ in range(args.measured_batches)]
        timing = {"status": "PASS", "environment": environment, "k": args.k, "window_target_ms": "20-50", "warmup_batches": args.warmup_batches, "measured_batches": args.measured_batches, "python_wall_us": stats(wall_samples), "cuda_event_us": stats(event_samples), "batched_outer_event_us_per_invocation": stats(batch_samples), "old_inner_event_us": {"median": 72.51200079917908, "source": "real_loop_002/boundary_timing.json", "known_perturbation": "inner CUDA Event around authentic callable"}, "measurement_method": "CUDA Event outer windows; no inner Event in batched estimate; perf_counter wall includes synchronization", "known_perturbation": "single-call Event and Python wall are overhead-sensitive; batched estimate amortizes timing boundary but still includes authentic launch/API path"}
        (args.out / "timing_attribution.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        (args.out / "batched_swiglu_timing.json").write_text(json.dumps({"status": "PASS", "k": args.k, "window_median_ms": statistics.median(batch_samples) * args.k / 1000.0, "per_invocation": stats(batch_samples), "raw_samples_us": batch_samples}, indent=2), encoding="utf-8")
        (args.out / "loop_003_runtime.json").write_text(json.dumps({"environment": environment, "footprint": footprint, "status": "PASS", "note": "full timing and footprint run"}, indent=2), encoding="utf-8")
    except Exception as exc:
        (args.out / "loop_003_runtime.json").write_text(json.dumps({"status": "BLOCKED", "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        return 0
    finally:
        if pg_here:
            parallel_state.destroy_model_parallel(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
