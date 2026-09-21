"""Phase 14-B source-grounded Megatron SwiGLU adapter.

The activation branch is executed by Megatron's exact MLP.forward. The linear
builders are deliberately local TP=1 adapters so the harness does not claim a
TE or distributed backend that is unavailable on the host.
"""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import TargetAdapter


ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = Path(__file__).resolve().parents[3] / "megatron-lm"
SPEC = ROOT / "targets" / "megatron_5be9626" / "swiglu.json"
CONTRACT = ROOT / "targets" / "megatron_5be9626" / "swiglu" / "replay_contract.json"


class _LocalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, **_kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, x):
        y = F.linear(x, self.weight, None)
        return y, self.bias


class MegatronSwiGLUAdapter(TargetAdapter):
    def __init__(self, device="cpu", dtype=torch.float32, shape=(4, 2, 16), ffn_hidden_size=32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.seq, self.micro_batch, self.hidden = shape
        self.ffn_hidden_size = ffn_hidden_size
        if str(MEGATRON_ROOT) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(MEGATRON_ROOT))
        try:
            from megatron.core.transformer.mlp import MLP, MLPSubmodules
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Megatron exact source could not be imported through its normal package path; "
                f"missing dependency: {exc.name}. This is a dependency blocker, not a replay fallback."
            ) from exc
        from megatron.core.transformer.transformer_config import TransformerConfig
        self.MLP = MLP
        config = TransformerConfig(
            num_layers=1, hidden_size=self.hidden, num_attention_heads=1,
            ffn_hidden_size=self.ffn_hidden_size, gated_linear_unit=True,
            activation_func=F.silu, add_bias_linear=False,
            bias_activation_fusion=False, use_te_activation_func=False,
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        )
        self.config = config
        submodules = MLPSubmodules(
            linear_fc1=lambda *args, **kw: _LocalLinear(args[0], args[1], **kw),
            linear_fc2=lambda *args, **kw: _LocalLinear(args[0], args[1], **kw),
        )
        self.module = MLP(config, submodules, input_size=self.hidden, ffn_hidden_size=self.ffn_hidden_size)
        self.module.to(device=self.device, dtype=self.dtype).eval()

    def load_target_spec(self):
        return json.loads(SPEC.read_text(encoding="utf-8"))

    def prepare_inputs(self, seed=0, dtype=None):
        dtype = dtype or self.dtype
        torch.manual_seed(seed)
        x = torch.randn(self.seq, self.micro_batch, self.hidden, device=self.device, dtype=dtype)
        return {"hidden_states": x}

    def run_megatron_reference(self, inputs):
        with torch.no_grad():
            output, output_bias = self.module(inputs["hidden_states"])
        return {"output": output, "output_bias": output_bias}

    def run_replay(self, inputs):
        # Independent oracle for the same activation boundary. FC1 is the
        # adapter module's output, then the exact source semantics are replayed.
        with torch.no_grad():
            intermediate, bias = self.module.linear_fc1(inputs["hidden_states"])
            if bias is not None:
                intermediate = intermediate + bias
            gate, up = torch.chunk(intermediate, 2, dim=-1)
            activated = F.silu(gate) * up
            output, output_bias = self.module.linear_fc2(activated)
        return {"output": output, "output_bias": output_bias}

    def compare_outputs(self, reference, replay):
        diff = (reference["output"] - replay["output"]).abs()
        return {"max_abs_error": float(diff.max().item()), "relative_error": float((diff / (reference["output"].abs() + 1e-8)).max().item()), "shape": list(reference["output"].shape), "dtype": str(reference["output"].dtype), "pass": bool(torch.allclose(reference["output"], replay["output"], atol=1e-6, rtol=1e-5))}

    def benchmark_reference(self, inputs, warmup=5, iterations=30):
        for _ in range(warmup):
            self.run_megatron_reference(inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iterations): self.run_megatron_reference(inputs)
            end.record(); torch.cuda.synchronize(self.device)
            return start.elapsed_time(end) * 1000.0 / iterations
        start = time.perf_counter()
        for _ in range(iterations): self.run_megatron_reference(inputs)
        return (time.perf_counter() - start) * 1e6 / iterations
