"""Phase 14-D stream-safe Episode 2 integration wrapper."""
import ctypes
from pathlib import Path
import torch
import torch.nn.functional as F

ATOL = 2e-3
RTOL = 2e-3

class SwigluIntegration:
    def __init__(self, library_path, official_shapes=((16,1,1024),(64,2,1024),(128,2,1024))):
        self.lib = ctypes.CDLL(str(library_path))
        self.lib.launch_swiglu_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
        self.lib.launch_swiglu_stream.restype = None
        self.backward_available = hasattr(self.lib, "launch_swiglu_backward_stream")
        if self.backward_available:
            self.lib.launch_swiglu_backward_stream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
            self.lib.launch_swiglu_backward_stream.restype = None
        self.official_shapes = set(tuple(x) for x in official_shapes)

    def validate(self, intermediate, bias=None, allow_shape=False):
        if not intermediate.is_cuda or intermediate.dtype != torch.float16:
            raise RuntimeError("REJECT_INTEGRATION: expected CUDA FP16 intermediate")
        if not intermediate.is_contiguous():
            raise RuntimeError("REJECT_INTEGRATION: intermediate must be contiguous")
        if intermediate.dim() != 2 or intermediate.shape[1] % 2:
            raise RuntimeError("REJECT_INTEGRATION: expected [rows, even_width]")
        if not allow_shape and not any(intermediate.shape[0] == s*b and intermediate.shape[1] == 8192 for s,b,_ in self.official_shapes):
            raise RuntimeError("REJECT_INTEGRATION: unsupported official shape")
        if bias is not None:
            if not bias.is_cuda or bias.dtype != torch.float16 or not bias.is_contiguous() or bias.numel() != intermediate.shape[1]:
                raise RuntimeError("REJECT_INTEGRATION: bias must be contiguous CUDA FP16 [width]")
            if bias.device != intermediate.device:
                raise RuntimeError("REJECT_INTEGRATION: bias device mismatch")

    def forward(self, intermediate, bias=None, offset=0.0, allow_shape=False):
        self.validate(intermediate, bias, allow_shape=allow_shape)
        out = torch.empty((intermediate.shape[0], intermediate.shape[1] // 2),
                          device=intermediate.device, dtype=intermediate.dtype)
        stream = torch.cuda.current_stream(intermediate.device).cuda_stream
        self.lib.launch_swiglu_stream(ctypes.c_void_p(intermediate.data_ptr()),
            ctypes.c_void_p(0 if bias is None else bias.data_ptr()), ctypes.c_void_p(out.data_ptr()),
            intermediate.shape[0], intermediate.shape[1], float(offset), ctypes.c_void_p(stream))
        return out

    def backward(self, intermediate, grad_output, offset=0.0, allow_shape=False):
        x = intermediate.reshape(-1, intermediate.shape[-1])
        go = grad_output.reshape(-1, grad_output.shape[-1])
        self.validate(x, None, allow_shape=True)
        if not self.backward_available or not go.is_cuda or go.dtype != torch.float16 or not go.is_contiguous():
            raise RuntimeError("REJECT_INTEGRATION: backward adapter unavailable or invalid grad_output")
        if tuple(go.shape) != (x.shape[0], x.shape[1] // 2):
            raise RuntimeError("REJECT_INTEGRATION: grad_output shape mismatch")
        out = torch.empty_like(x)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        self.lib.launch_swiglu_backward_stream(x.data_ptr(), go.data_ptr(), out.data_ptr(), x.shape[0], x.shape[1], float(offset), stream)
        return out.reshape_as(intermediate)

class SwigluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, adapter, intermediate, bias, offset, allow_shape):
        out = adapter.forward(intermediate, bias, offset, allow_shape)
        ctx.save_for_backward(intermediate, bias if bias is not None else torch.tensor([], device=intermediate.device, dtype=intermediate.dtype))
        ctx.has_bias = bias is not None
        ctx.offset = float(offset)
        ctx.adapter = adapter
        return out

    @staticmethod
    def backward(ctx, grad_out):
        intermediate, saved_bias = ctx.saved_tensors
        x = intermediate if not ctx.has_bias else intermediate + saved_bias
        gate, up = torch.chunk(x, 2, dim=-1)
        sig = torch.sigmoid(gate)
        silu = gate * sig
        dsilu = sig * (1.0 + gate * (1.0 - sig))
        grad_up = grad_out * silu
        grad_gate = grad_out * (up + ctx.offset) * dsilu
        if getattr(ctx.adapter, "backward_available", False) and not ctx.has_bias:
            grad_intermediate = ctx.adapter.backward(intermediate, grad_out, ctx.offset, allow_shape=True)
        else:
            grad_intermediate = torch.cat((grad_gate, grad_up), dim=-1)
        grad_bias = grad_intermediate.sum(dim=0) if ctx.has_bias else None
        return None, grad_intermediate, grad_bias, None, None

def apply_swiglu(adapter, intermediate, bias=None, offset=0.0, allow_shape=False):
    return SwigluFunction.apply(adapter, intermediate, bias, offset, allow_shape)
