"""Read-only diagnostic for the authentic Megatron fused SwiGLU path."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
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
    import torch.nn as nn
    import torch.nn.functional as F
    import triton
    import megatron.core.fusions.fused_bias_swiglu as fused
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    class LocalLinear(nn.Module):
        def __init__(self, input_size, output_size, config=None, bias=True, **_kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(output_size, input_size) * 0.02)
            self.bias = nn.Parameter(torch.randn(output_size) * 0.01) if bias else None

        def forward(self, hidden_states):
            return F.linear(hidden_states, self.weight), self.bias

    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, ffn_hidden_size=16,
        tensor_model_parallel_size=1, sequence_parallel=False, gated_linear_unit=True,
        activation_func=F.silu, bias_activation_fusion=True, add_bias_linear=True,
        params_dtype=torch.float32, use_cpu_initialization=True,
        perform_initialization=False, transformer_impl="local",
    )
    model = MLP(config, MLPSubmodules(linear_fc1=LocalLinear, linear_fc2=LocalLinear)).to("cuda")
    x = torch.randn(2, 3, 8, device="cuda")
    lines = []
    driver_path = args.repo.resolve() / ".runtime_deps" / "triton" / "backends" / "nvidia" / "driver.c"
    lines.append("AUTHENTIC_SWIGLU_DIAGNOSTIC")
    lines.append(f"interpreter={sys.executable}")
    lines.append(f"python={sys.version}")
    lines.append(f"platform={platform.platform()}")
    lines.append(f"torch={torch.__version__}")
    lines.append(f"torch_cuda={torch.version.cuda}")
    lines.append(f"cuda_available={torch.cuda.is_available()}")
    lines.append(f"device={torch.cuda.get_device_name(0)}")
    lines.append(f"device_capability={torch.cuda.get_device_capability(0)}")
    lines.append(f"triton={triton.__version__}")
    lines.append(f"triton_file={triton.__file__}")
    lines.append(f"fused_file={fused.__file__}")
    lines.append(f"driver_c={driver_path}")
    if driver_path.exists():
        source_lines = driver_path.read_text(encoding="utf-8", errors="replace").splitlines()
        lines.append("driver_c_lines_1079_1087:")
        for number in range(1079, 1088):
            if number <= len(source_lines):
                lines.append(f"{number}: {source_lines[number - 1]}")
    try:
        cl = subprocess.run(["cl.exe"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        lines.append("cl_version_output:")
        lines.append(cl.stdout or "")
        lines.append(cl.stderr or "")
    except Exception:
        lines.append("cl_version_error:")
        lines.append(traceback.format_exc())
    lines.append("environment_paths:")
    for key in ("PATH", "INCLUDE", "LIB", "LIBPATH", "CUDA_PATH", "WindowsSdkDir", "VCToolsInstallDir"):
        lines.append(f"{key}={os.environ.get(key, '<unset>')}")
    lines.append("original_call:")
    lines.append("megatron.core.transformer.mlp.MLP.forward -> bias_swiglu_impl")
    try:
        torch.manual_seed(20260922)
        output = model(x)
        torch.cuda.synchronize()
        lines.append(f"result=PASS shape={tuple(output[0].shape if isinstance(output, tuple) else output.shape)}")
    except Exception as exc:
        lines.append(f"result=BLOCKED {type(exc).__name__}: {exc}")
        lines.append("full_traceback:")
        lines.append(traceback.format_exc())
    (args.out / "original_failure.txt").write_text("\n".join(str(item) for item in lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
