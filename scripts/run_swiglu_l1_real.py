"""Execute one real Megatron MLP SwiGLU L1 correctness run.

This script imports Megatron read-only, builds the smallest real MLP.forward
path with local torch linear builders, and patches only the imported D-side
process symbol ``megatron.core.transformer.mlp.bias_swiglu_impl``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / ".runtime_deps"))
    sys.path.insert(1, str(repo))
    sys.path.insert(2, str(args.megatron_root.resolve()))

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from lab.runtime.evaluators.integration_oj import IntegrationOJ
    from lab.runtime.integrations.swiglu_l1 import (
        SwiGLUIntegrationResult,
        SwiGLUL1Harness,
        _error,
        build_reasoner_snapshot,
        build_swiglu_oj_evidence,
        file_sha256,
        write_result,
    )
    import megatron.core.transformer.mlp as megatron_mlp
    import megatron.core.fusions.fused_bias_swiglu as fused_swiglu
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    # The source-declared baseline uses @jit_fuser.  Disable only the three
    # process-local callable globals it invokes so Windows correctness can run
    # without compiling a Triton/Inductor kernel.  Megatron source is untouched.
    def plain_swiglu(values):
        gate, up = torch.chunk(values, 2, dim=-1)
        return F.silu(gate) * up

    def plain_bias_swiglu(values, bias):
        return plain_swiglu(values + bias)

    def plain_swiglu_back(grad, values):
        gate, up = torch.chunk(values, 2, dim=-1)
        sigmoid = torch.sigmoid(gate)
        return torch.cat((grad * sigmoid * (1 + gate * (1 - sigmoid)) * up, grad * F.silu(gate)), dim=-1)

    def plain_bias_swiglu_back(grad, values, bias):
        return plain_swiglu_back(grad, values + bias)

    fused_swiglu.swiglu = plain_swiglu
    fused_swiglu.bias_swiglu = plain_bias_swiglu
    fused_swiglu.swiglu_back = plain_swiglu_back
    fused_swiglu.bias_swiglu_back = plain_bias_swiglu_back

    class LocalLinear(nn.Module):
        def __init__(self, input_size, output_size, config=None, bias=True, **_kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(output_size, input_size) * 0.02)
            self.bias = nn.Parameter(torch.randn(output_size) * 0.01) if bias else None

        def forward(self, hidden_states):
            return F.linear(hidden_states, self.weight), self.bias

    def make_model():
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            ffn_hidden_size=16,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=True,
            add_bias_linear=True,
            params_dtype=torch.float32,
            use_cpu_initialization=True,
            perform_initialization=False,
            transformer_impl="local",
        )
        return MLP(config, MLPSubmodules(linear_fc1=LocalLinear, linear_fc2=LocalLinear))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260922)
    baseline_model = make_model().to(device)
    candidate_model = make_model().to(device)
    candidate_model.load_state_dict(baseline_model.state_dict())
    input_base = torch.randn(2, 3, 8, device=device, dtype=torch.float32)
    grad_output = torch.randn(2, 3, 8, device=device, dtype=torch.float32)

    baseline_input = input_base.detach().clone().requires_grad_(True)
    baseline_output_raw = baseline_model(baseline_input)
    baseline_output = baseline_output_raw[0] if isinstance(baseline_output_raw, tuple) else baseline_output_raw
    (baseline_output * grad_output).sum().backward()
    baseline_input_grad = baseline_input.grad.detach().clone()
    baseline_parameter_grads = {name: param.grad.detach().clone() for name, param in baseline_model.named_parameters() if param.grad is not None}

    def reference_swiglu(input_tensor, bias, *_args, **_kwargs):
        original_shape = input_tensor.shape
        flattened = input_tensor.reshape(-1, original_shape[-1])
        if bias is not None:
            flattened = flattened + bias
        gate, up = torch.chunk(flattened, 2, dim=-1)
        output = F.silu(gate) * up
        return output.reshape(*original_shape[:-1], output.shape[-1])

    candidate_input = input_base.detach().clone().requires_grad_(True)
    with SwiGLUL1Harness(megatron_mlp) as harness:
        replacement = harness.install(reference_swiglu)
        start = time.perf_counter()
        candidate_output_raw = candidate_model(candidate_input)
        candidate_output = candidate_output_raw[0] if isinstance(candidate_output_raw, tuple) else candidate_output_raw
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        (candidate_output * grad_output).sum().backward()
        candidate_input_grad = candidate_input.grad.detach().clone()
        candidate_parameter_grads = {name: param.grad.detach().clone() for name, param in candidate_model.named_parameters() if param.grad is not None}
        forward = _error(baseline_output.detach(), candidate_output.detach())
        backward_input = _error(baseline_input_grad, candidate_input_grad)
        parameter_errors = {name: _error(baseline_parameter_grads[name], candidate_parameter_grads[name]) for name in baseline_parameter_grads}
        gradient_checks = [backward_input, *parameter_errors.values()]
        backward = {
            "status": "CHECKED",
            "input": backward_input,
            "parameters": parameter_errors,
            "max_abs_grad_error": max(item["max_abs_error"] for item in gradient_checks),
            "max_rel_grad_error": max(item["max_rel_error"] for item in gradient_checks),
            "finite": all(item["finite"] for item in gradient_checks),
        }
        captured = dict(replacement.calls[-1])
        captured["input"]["contiguous"] = bool(candidate_input.is_contiguous())
        captured["output"]["contiguous"] = bool(candidate_output.is_contiguous())
        captured["parameter_shapes"] = {name: list(param.shape) for name, param in candidate_model.named_parameters()}
        captured["tensor_parallel"] = {"size": 1, "rank": 0, "group": "local-single-rank"}
        captured["sequence_parallel"] = {"enabled": False, "leading_shape": list(candidate_input.shape[:2])}

    contract = repo / "targets/megatron_5be9626/integration_contracts/swiglu.json"
    candidate_source = repo / "lab/runtime/integrations/swiglu_l1.py"
    candidate_hash = file_sha256(candidate_source)
    contract_hash = file_sha256(contract)
    result = SwiGLUIntegrationResult(
        source_commit="5be9626709af2722333bf54797c954c09edeada3",
        integration_contract_hash=contract_hash,
        candidate_hash=candidate_hash,
        captured_shape=captured,
        forward_correctness=forward,
        backward_correctness=backward,
        replacement_invocations=replacement.invocation_count,
        baseline_target_invocations=1,
        fallback_detected=replacement.invocation_count == 0,
        runtime_environment={
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "distributed_invariants": True,
            "tensor_parallel_size": 1,
            "sequence_parallel": False,
            "megatron_mlp": str(Path(megatron_mlp.__file__).resolve()),
        },
        timing_if_available={"module_ms": elapsed_ms},
        status="INTEGRATION_PASS",
    )
    evidence = build_swiglu_oj_evidence(result)
    judged = IntegrationOJ(lambda _: evidence).evaluate({
        "expected_replacement_marker": "aka-local-swiglu-reference-v1",
        "required_metrics": ["module_ms"],
        "expected_contract_hash": contract_hash,
        "expected_candidate_hash": candidate_hash,
    })
    result.status = "INTEGRATION_PASS" if judged.verdict == "INTEGRATION_PASS" else "INTEGRATION_FAIL"
    result.evidence = {"l1_oj": judged.to_dict() if hasattr(judged, "to_dict") else {"verdict": judged.verdict, "checks": judged.checks, "reasons": judged.reasons}}
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    write_result(args.artifact_root / "swiglu_integration_result.json", result)
    (args.artifact_root / "l1_oj_result.json").write_text(json.dumps(result.evidence, indent=2, sort_keys=True), encoding="utf-8")
    reasoner = build_reasoner_snapshot(captured)
    (args.artifact_root / "real_performance_facts.json").write_text(json.dumps(reasoner["performance_facts"], indent=2, sort_keys=True), encoding="utf-8")
    (args.artifact_root / "ranked_opportunities.json").write_text(json.dumps(reasoner["top_opportunities"], indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result.status, "l1_verdict": judged.verdict, "captured_shape": captured, "forward": forward, "backward": backward, "replacement_invocations": replacement.invocation_count, "artifact_root": str(args.artifact_root.resolve())}, indent=2, default=str))
    return 0 if result.status == "INTEGRATION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
