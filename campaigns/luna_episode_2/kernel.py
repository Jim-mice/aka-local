import os
from pathlib import Path
import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load

cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")

_ROOT = Path(__file__).resolve().parent
_ext = load(
    name="aka_local_swiglu_v2b_sm120_ptxas2",
    sources=[str(_ROOT / "swiglu_kernel_v2b.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_120", "-Xcompiler=/Zc:preprocessor", "--ptxas-options=-v"],
    verbose=True,
)


class Model(torch.nn.Module):
    def __init__(self, rows: int, cols: int, dtype: str = "float16"):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.dtype = getattr(torch, dtype)

    def forward(self, gate, up):
        out = torch.empty_like(gate)
        _ext.launch_swiglu(gate, up, out)
        return out
