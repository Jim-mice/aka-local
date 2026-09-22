"""Build source-grounded SwiGLU facts and run the existing planner once."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--megatron-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo.resolve() / ".runtime_deps"))
    sys.path.insert(1, str(args.repo.resolve()))
    sys.path.insert(2, str(args.megatron_root.resolve()))

    import megatron.core.fusions.fused_bias_swiglu as fused
    import megatron.core.transformer.mlp as mlp
    from lab.runtime.reasoning.hypothesis_planner import GenerativeHypothesisPlanner
    from lab.runtime.reasoning.mechanism_memory import MechanismStore
    from lab.runtime.reasoning.performance_model import PerformanceFacts

    source_files = {
        "mlp.py": args.megatron_root / "megatron/core/transformer/mlp.py",
        "fused_bias_swiglu.py": args.megatron_root / "megatron/core/fusions/fused_bias_swiglu.py",
    }
    source_evidence = {
        "commit": "5be9626709af2722333bf54797c954c09edeada3",
        "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in source_files.items()},
        "symbols": {
            "MLP.forward": {"source": inspect.getsource(mlp.MLP.forward), "module": mlp.__file__},
            "bias_swiglu_impl": {"source": inspect.getsource(fused.bias_swiglu_impl), "module": fused.__file__},
            "BiasSwiGLUFunction.forward": {"source": inspect.getsource(fused.BiasSwiGLUFunction.forward), "module": fused.__file__},
            "BiasSwiGLUFunction.backward": {"source": inspect.getsource(fused.BiasSwiGLUFunction.backward), "module": fused.__file__},
            "bias_swiglu": {"source": inspect.getsource(fused.bias_swiglu), "module": fused.__file__},
        },
        "source_derived_dataflow": {
            "fc1_output": "MLP.forward receives intermediate_parallel and bias_parallel from linear_fc1.",
            "bias_handling": "With bias_activation_fusion and gated SiLU, MLP.forward calls bias_swiglu_impl(intermediate_parallel, bias_parallel, ...).",
            "activation_boundary": "bias_swiglu_impl reshapes input, calls BiasSwiGLUFunction.apply, and restores the original rank before returning.",
            "activation_function": "BiasSwiGLUFunction.forward saves input_for_backward and bias, then calls bias_swiglu(input, bias); bias_swiglu performs bias addition and swiglu.",
            "fc2_consumer": "MLP.forward passes the returned intermediate_parallel to linear_fc2 in a subsequent module call.",
            "backward": "BiasSwiGLUFunction.backward reads saved input and bias, applies bias_swiglu_back, and returns gradients for input and bias.",
            "real_materialized_values": ["linear_fc1 output tensor", "linear_fc1 bias tensor", "SwiGLU activation output consumed by linear_fc2", "saved input and bias for custom autograd backward"],
            "python_only_boundaries": ["bias_swiglu_impl wrapper/reshape", "MLP.forward Python module calls"],
            "known_kernel_boundary": "The source shows fused bias+SwiGLU callable followed by a separate linear_fc2 module call; exact GPU launch and HBM behavior are not established by this source inspection.",
        },
    }
    (args.out / "authentic_dataflow.json").write_text(json.dumps(source_evidence, indent=2, ensure_ascii=False), encoding="utf-8")

    facts = PerformanceFacts(
        operator="swiglu",
        shape={"batch": 2, "sequence": 3, "hidden": 8, "fc1_output": 32, "value_dtype": "torch.float32"},
        dtype="torch.float32",
        dataflow=("linear_fc1 -> intermediate_parallel,bias_parallel", "bias_swiglu_impl -> BiasSwiGLUFunction", "bias_swiglu -> swiglu", "activation_output -> linear_fc2"),
        tensor_lifetimes={"linear_fc1_output": "bias_swiglu forward and custom backward", "linear_fc1_bias": "bias_swiglu forward and custom backward", "activation_output": "returned from fused activation and consumed by linear_fc2"},
        global_memory_reads=("linear_fc1_output", "linear_fc1_bias", "activation_output_by_linear_fc2"),
        global_memory_writes=("activation_output",),
        reductions=(),
        synchronizations=(),
        kernel_launches=None,
        producer_consumer_boundaries=("authentic_bias_swiglu_output -> linear_fc2",),
        known_reuse=("saved linear_fc1 input and bias are reused by custom backward",),
        fixed_dimensions={"hidden": 8, "fc1_output": 32, "value_width": 16},
        dynamic_dimensions=("batch", "sequence"),
        parallel_mapping={"tensor_parallel": "1", "sequence_parallel": "false", "layout": "[sequence,batch,hidden] contiguous capture"},
        profile_evidence={
            "evidence_status": "MEASURED_PLUS_SOURCE_DERIVED",
            "shape_capture": "artifacts/integration/swiglu/real_l1/swiglu_integration_result.json",
            "authentic_boundary_latency_ms": {"value": 0.09344, "evidence": "MEASURED historical authentic profile median"},
            "local_model_split_fraction": {"value": 0.015571342909175317, "evidence": "MEASURED_CAMPAIGN_MEDIAN_SPLIT"},
            "local_model_paired_fraction": {"value": 0.041956971532107534, "evidence": "MEASURED_CAMPAIGN_MEDIAN_PAIRED_INSTRUMENTED"},
            "register_pressure": {"value": None, "evidence": "UNKNOWN"},
            "shared_memory": {"value": None, "evidence": "UNKNOWN"},
            "occupancy": {"value": None, "evidence": "UNKNOWN"},
            "kernel_count": {"value": None, "evidence": "UNKNOWN"},
            "bytes": {"value": None, "evidence": "UNKNOWN"},
            "supports_mechanism_ids": [], "contradicts_mechanism_ids": [], "idle_resources": [],
        },
        unknown_fields=("global_memory_bytes", "kernel_launch_count", "registers_per_thread", "shared_memory_bytes", "occupancy", "CUPTI_kernel_breakdown", "backward_kernel_breakdown", "activation_HBM_round_trip"),
        e2e_profile={"scope": "LOCAL_MINIMAL_GPTMODEL", "operator_fraction_of_step": 0.015571342909175317, "paired_fraction_instrumentation_affected": 0.041956971532107534, "max_possible_e2e_speedup_split": 1.015817644881643, "nine_grid": None},
    )
    (args.out / "performance_facts_v5.json").write_text(json.dumps(facts.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    mechanism_store = MechanismStore(args.repo / "knowledge/mechanisms.jsonl")
    matched = mechanism_store.query("swiglu", operator="swiglu")
    planning = GenerativeHypothesisPlanner().plan(facts, matched)
    result = planning.to_dict()
    result["source_commit"] = source_evidence["commit"]
    result["matched_mechanism_ids"] = [item.mechanism_id for item in matched]
    result["facts_artifact"] = str(args.out / "performance_facts_v5.json")
    (args.out / "hypotheses.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"matched_mechanisms": len(matched), "raw": len(result["raw_generation"]), "validated": len(result["validated"]), "ranked": len(result["ranked"]), "rejected": len(result["rejected"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
