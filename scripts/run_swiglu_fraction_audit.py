"""Audit authentic SwiGLU fraction with batched CUDA-event campaigns.

This is measurement-only. It does not change Megatron source or qualification policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import traceback


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "mean": None, "stdev": None, "cv": None, "min": None, "max": None, "p10": None, "p90": None}
    ordered = sorted(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "median": statistics.median(values), "mean": mean, "stdev": sd, "cv": sd / mean if mean else None, "min": min(values), "max": max(values), "p10": ordered[max(0, int(len(ordered) * .1) - 1)], "p90": ordered[min(len(ordered) - 1, int(len(ordered) * .9))]}


def robust(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "mad": None, "iqr": None, "p10": None, "p90": None}
    ordered = sorted(values)
    med = statistics.median(ordered)
    deviations = [abs(x - med) for x in ordered]
    q1, q3 = statistics.quantiles(ordered, n=4, method="inclusive")[0], statistics.quantiles(ordered, n=4, method="inclusive")[2] if len(ordered) >= 2 else (med, med)
    return {"median": med, "mad": statistics.median(deviations), "iqr": q3 - q1, "p10": ordered[max(0, int(len(ordered) * .1) - 1)], "p90": ordered[min(len(ordered) - 1, int(len(ordered) * .9))]}


def bootstrap_ci(values: list[float], seed: int = 20260922, draws: int = 2000) -> list[float | None]:
    if len(values) < 2:
        return [None, None]
    rng = random.Random(seed)
    medians = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        medians.append(statistics.median(sample))
    medians.sort()
    return [medians[int(.025 * draws)], medians[int(.975 * draws)]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--megatron-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--campaigns", type=int, default=5)
    ap.add_argument("--warmup-batches", type=int, default=20)
    ap.add_argument("--measured-batches", type=int, default=30)
    ap.add_argument("--steps-per-batch", type=int, default=32)
    ap.add_argument("--boundary-invocations-per-batch", type=int, default=512)
    args = ap.parse_args()
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
        (args.out / "fraction_audit.json").write_text(json.dumps({"status": "BLOCKED", "reason": "CUDA unavailable"}, indent=2), encoding="utf-8")
        return 0
    device = torch.device("cuda")
    pg = False
    init_file = (args.out / "fraction_audit_pg_init").resolve()
    try:
        if init_file.exists():
            init_file.unlink()
        dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file:///{init_file.as_posix()}")
        pg = True
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1, expert_model_parallel_size=1, create_gloo_process_groups=True)
        initialize_rng_tracker(force_reset=True)
        get_cuda_rng_tracker().add("model-parallel-rng", 20260922)
    except Exception as exc:
        (args.out / "fraction_audit.json").write_text(json.dumps({"status": "BLOCKED", "reason": repr(exc), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        return 0

    seq, batch, hidden = 3, 2, 8
    mask = torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1).view(1, 1, seq, seq)
    token_ids = torch.randint(0, 32, (batch, seq), device=device)

    def config():
        return TransformerConfig(num_layers=1, hidden_size=hidden, num_attention_heads=2, ffn_hidden_size=16, tensor_model_parallel_size=1, sequence_parallel=False, gated_linear_unit=True, activation_func=F.silu, bias_activation_fusion=True, add_bias_linear=True, params_dtype=torch.float32, use_cpu_initialization=True, perform_initialization=False, transformer_impl="local", hidden_dropout=0.0, attention_dropout=0.0, attention_softmax_in_fp32=False, bias_dropout_fusion=False)

    def output(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                if torch.is_tensor(item):
                    return item
        raise TypeError(type(value).__name__)

    def factories():
        def layer():
            return build_module(get_gpt_layer_local_spec(normalization="RMSNorm"), config=config(), layer_number=1).to(device).train(), "layer"
        def block():
            c = config()
            return TransformerBlock(c, get_gpt_decoder_block_spec(c, use_transformer_engine=False, normalization="RMSNorm")).to(device).train(), "block"
        def model():
            c = config()
            return GPTModel(c, get_gpt_layer_local_spec(normalization="RMSNorm"), vocab_size=32, max_sequence_length=seq, position_embedding_type="none").to(device).train(), "model"
        return {"TRANSFORMER_LAYER": layer, "TRANSFORMER_BLOCK": block, "MINIMAL_MODEL": model}

    def call(module, kind, tensor):
        if kind == "layer":
            return output(module(tensor, attention_mask=mask)[0])
        if kind == "block":
            return output(module(tensor, attention_mask=mask))
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        return output(module(token_ids, positions, mask))

    def one_step(module, kind, optimizer, record=False):
        optimizer.zero_grad(set_to_none=True)
        tensor = (torch.randn(seq, batch, hidden, device=device) if kind != "model" else token_ids)
        if kind != "model":
            tensor.requires_grad_(True)
        result = call(module, kind, tensor)
        (result.square().mean()).backward()
        optimizer.step()

    original = megatron_mlp.bias_swiglu_impl
    all_results: dict[str, object] = {}
    try:
        for scope, factory in factories().items():
            scope_result: dict[str, object] = {"scope": scope, "campaigns": [], "steps_per_batch": args.steps_per_batch, "target_window_ms": 50.0, "measurement_status": "MEASUREMENT_COMPLETE"}
            for campaign in range(1, args.campaigns + 1):
                torch.manual_seed(20260922 + campaign)
                module, kind = factory()
                optimizer = torch.optim.SGD(module.parameters(), lr=0.0)
                captured: list[tuple[tuple[object, ...], dict[str, object]]] = []
                def capture_boundary(*bargs, **bkwargs):
                    if not captured:
                        captured.append((tuple(x.detach() if torch.is_tensor(x) else x for x in bargs), dict(bkwargs)))
                    return original(*bargs, **bkwargs)
                megatron_mlp.bias_swiglu_impl = capture_boundary
                one_step(module, kind, optimizer)
                torch.cuda.synchronize()
                megatron_mlp.bias_swiglu_impl = original

                for _ in range(args.warmup_batches):
                    for _ in range(args.steps_per_batch):
                        one_step(module, kind, optimizer)
                torch.cuda.synchronize()

                # A: uninstrumented whole-step batched windows.
                baseline_batch_events = []
                for _ in range(args.measured_batches):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.steps_per_batch):
                        one_step(module, kind, optimizer)
                    end.record(); baseline_batch_events.append((start, end))
                torch.cuda.synchronize()
                baseline_batch_ms = [float(a.elapsed_time(b)) for a, b in baseline_batch_events]

                # B/D: inner boundary instrumentation and paired step identities.
                paired_step_events = []
                paired_boundary_events = []
                batch_events = []
                current: list[tuple[object, object]] | None = None
                def timed_boundary(*bargs, **bkwargs):
                    nonlocal current
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); value = original(*bargs, **bkwargs); end.record()
                    if current is not None:
                        current.append((start, end))
                    return value
                megatron_mlp.bias_swiglu_impl = timed_boundary
                for _ in range(args.measured_batches):
                    bs, be = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    bs.record()
                    for _ in range(args.steps_per_batch):
                        ws, we = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        current = []
                        ws.record(); one_step(module, kind, optimizer); we.record()
                        paired_step_events.append((ws, we)); paired_boundary_events.append(current)
                        current = None
                    be.record(); batch_events.append((bs, be))
                torch.cuda.synchronize()
                instrumented_batch_ms = [float(a.elapsed_time(b)) for a, b in batch_events]
                paired_whole = [float(a.elapsed_time(b)) for a, b in paired_step_events]
                paired_boundary = [sum(float(a.elapsed_time(b)) for a, b in events) for events in paired_boundary_events]
                megatron_mlp.bias_swiglu_impl = original

                # C: independent batched authentic boundary microbenchmark.
                if not captured:
                    raise RuntimeError("could not capture authentic boundary arguments")
                bargs, bkwargs = captured[0]
                micro_events = []
                for _ in range(args.measured_batches):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.boundary_invocations_per_batch):
                        original(*bargs, **bkwargs)
                    end.record(); micro_events.append((start, end))
                torch.cuda.synchronize()
                micro_batch_ms = [float(a.elapsed_time(b)) for a, b in micro_events]
                campaign_result = {
                    "campaign": campaign,
                    "baseline_uninstrumented_batch_ms": baseline_batch_ms,
                    "instrumented_batch_ms": instrumented_batch_ms,
                    "paired_step_whole_ms": paired_whole,
                    "paired_step_boundary_ms": paired_boundary,
                    "boundary_micro_batch_ms": micro_batch_ms,
                    "baseline_per_step_ms": [x / args.steps_per_batch for x in baseline_batch_ms],
                    "instrumented_per_step_ms": [x / args.steps_per_batch for x in instrumented_batch_ms],
                    "boundary_micro_per_invocation_ms": [x / args.boundary_invocations_per_batch for x in micro_batch_ms],
                }
                scope_result["campaigns"].append(campaign_result)
            all_results[scope] = scope_result
    finally:
        megatron_mlp.bias_swiglu_impl = original
        if pg:
            parallel_state.destroy_model_parallel(); dist.destroy_process_group()

    for scope, result in all_results.items():
        campaigns = result["campaigns"]
        a = [statistics.median(x["baseline_per_step_ms"]) for x in campaigns]
        b = [statistics.median(x["instrumented_per_step_ms"]) for x in campaigns]
        micro = [statistics.median(x["boundary_micro_per_invocation_ms"]) for x in campaigns]
        paired = [statistics.median([u / w for u, w in zip(x["paired_step_boundary_ms"], x["paired_step_whole_ms"]) if w > 0]) for x in campaigns]
        split = [u / w for u, w in zip(micro, a)]
        result["campaign_summary"] = {"A_uninstrumented_whole_per_step_ms": robust(a), "B_instrumented_whole_per_step_ms": robust(b), "C_boundary_per_invocation_ms": robust(micro), "D_paired_fraction": robust(paired), "split_fraction_by_campaign": robust(split), "split_fraction_95ci": bootstrap_ci(split), "paired_fraction_95ci": bootstrap_ci(paired), "instrumentation_overhead_percent": (statistics.median(b) / statistics.median(a) - 1.0) * 100.0}
        result["fraction_estimators"] = {"f_split": statistics.median(split), "f_paired": statistics.median(paired), "f_split_ci95": bootstrap_ci(split), "f_paired_ci95": bootstrap_ci(paired)}
    payload = {"status": "PASS", "protocol": {"campaigns": args.campaigns, "warmup_batches": args.warmup_batches, "measured_batches": args.measured_batches, "steps_per_batch": args.steps_per_batch, "boundary_invocations_per_batch": args.boundary_invocations_per_batch, "window_selection_basis": "32 model steps target approximately 50 ms from prior ~2 ms step evidence; 512 boundary invocations target approximately 50 ms from prior ~0.09 ms boundary evidence", "cuda_event": True, "instrumentation_overhead_audited": True, "paired_identity": True}, "scopes": all_results}
    (args.out / "fraction_audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({scope: item["fraction_estimators"] for scope, item in all_results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
