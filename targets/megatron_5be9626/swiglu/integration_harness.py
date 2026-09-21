import argparse, ctypes, json, math, os, statistics, subprocess
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

ROOT = Path(__file__).resolve().parent
SHAPES = [(16,1,1024), (64,2,1024), (128,2,1024)]
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
BACKWARD_LIB = None

class LocalLinear(nn.Module):
    def __init__(self, ins, outs, **kw):
        super().__init__(); self.weight = nn.Parameter(torch.randn(outs, ins, device='cuda', dtype=torch.float16) * 0.01)
    def forward(self, x): return F.linear(x, self.weight), None

def make_mlp(fused):
    cfg = TransformerConfig(num_layers=1, hidden_size=1024, num_attention_heads=16,
        ffn_hidden_size=4096, gated_linear_unit=True, activation_func=F.silu,
        add_bias_linear=False, bias_activation_fusion=fused, use_te_activation_func=False,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    subs = MLPSubmodules(linear_fc1=lambda *a, **k: LocalLinear(a[0], a[1]),
                         linear_fc2=lambda *a, **k: LocalLinear(a[0], a[1]))
    return MLP(cfg, subs, input_size=1024, ffn_hidden_size=4096).to(device='cuda', dtype=torch.float16).train()

def load_adapter():
    so = os.environ['AKA_SWIGLU_SO']; lib = ctypes.CDLL(so)
    lib.launch_swiglu_stream.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p]
    return lib

def load_backward():
    global BACKWARD_LIB
    if BACKWARD_LIB is not None: return BACKWARD_LIB
    lib = ctypes.CDLL(os.environ['AKA_SWIGLU_BACKWARD_SO'])
    lib.launch_swiglu_backward_stream.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p]
    BACKWARD_LIB = lib
    return lib

def optimized_fn(lib, intermediate, bias, output):
    intermediate = intermediate.reshape(-1, intermediate.shape[-1])
    output = output.reshape(-1, output.shape[-1])
    stream = torch.cuda.current_stream(intermediate.device).cuda_stream
    lib.launch_swiglu_stream(intermediate.data_ptr(), 0 if bias is None else bias.data_ptr(), output.data_ptr(), intermediate.shape[0], intermediate.shape[1], 0.0, stream)
    return output

class SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lib, x, bias, offset):
        out = torch.empty((*x.shape[:-1], x.shape[-1]//2), device=x.device, dtype=x.dtype)
        optimized_fn(lib, x, bias, out)
        ctx.save_for_backward(x, bias if bias is not None else torch.tensor([], device=x.device, dtype=x.dtype)); ctx.has_bias=bias is not None
        ctx.offset=float(offset)
        return out
    @staticmethod
    def backward(ctx, go):
        x,b=ctx.saved_tensors; z=x if not ctx.has_bias else x+b; gate,up=torch.chunk(z,2,-1)
        sig=torch.sigmoid(gate); silu=gate*sig; ds=sig*(1+gate*(1-sig)); grad_gate=go*(up+ctx.offset)*ds; grad_up=go*silu
        if not ctx.has_bias and os.environ.get('AKA_SWIGLU_BACKWARD_SO'):
            gi=torch.empty_like(x); x2=x.reshape(-1,x.shape[-1]); go2=go.reshape(-1,go.shape[-1]); g2=gi.reshape(-1,gi.shape[-1]); blib=load_backward(); blib.launch_swiglu_backward_stream(x2.data_ptr(),go2.data_ptr(),g2.data_ptr(),x2.shape[0],x2.shape[1],ctx.offset,torch.cuda.current_stream(x.device).cuda_stream)
        else: gi=torch.cat((grad_gate, grad_up),-1)
        return None, gi, gi.sum(dim=tuple(range(gi.dim()-1))) if ctx.has_bias else None, None

def run(mode):
    lib = load_adapter(); torch.manual_seed(14)
    original, integrated = make_mlp(False), make_mlp(True); integrated.load_state_dict(original.state_dict())
    from megatron.core.transformer import mlp as mlp_mod
    old = mlp_mod.bias_swiglu_impl
    def sidecar(intermediate_parallel, bias_parallel, *args):
        return SwiGLUFn.apply(lib, intermediate_parallel, bias_parallel, 0.0)
    mlp_mod.bias_swiglu_impl = sidecar
    rows=[]
    try:
      for S,B,H in SHAPES:
        torch.manual_seed(100+S); x=torch.randn(S,B,H,device='cuda',dtype=torch.float16)
        with torch.no_grad():
          if mode == 'original':
            ro,_=original(x); ri=ro
          elif mode == 'profile':
            ri,_=integrated(x); ro=ri
          else:
            ro,_=original(x); ri,_=integrated(x)
          e=(ro-ri).abs(); rr=(e/(ro.abs()+1e-3)).max()
        row={'shape':[S,B,H],'forward_max_abs':float(e.max()),'forward_max_rel':float(rr),'forward_pass':bool(torch.allclose(ro,ri,atol=2e-3,rtol=2e-3))}
        if mode in ('original','benchmark','profile') and [S,B,H]==[128,2,1024]:
          fn = original if mode=='original' else integrated
          for _ in range(10): fn(x)
          torch.cuda.synchronize(); samples=[]
          for _ in range(5):
            a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
            for _ in range(100): fn(x)
            b.record(); torch.cuda.synchronize(); samples.append(a.elapsed_time(b)*1000/100)
          row['forward_samples_us']=samples; row['forward_mean_us']=statistics.mean(samples); row['forward_std_us']=statistics.stdev(samples); row['forward_cv']=row['forward_std_us']/row['forward_mean_us']
        if mode=='grad' and [S,B,H]==[128,2,1024]:
          xo=x.detach().requires_grad_(); xi=x.detach().requires_grad_(); go=torch.randn_like(xo)
          original.zero_grad(set_to_none=True); integrated.zero_grad(set_to_none=True)
          oo,_=original(xo); oi,_=integrated(xi); oo.backward(go); oi.backward(go)
          input_err=(xo.grad-xi.grad).abs(); row['input_grad_max_abs']=float(input_err.max()); row['input_grad_max_rel']=float((input_err/(xo.grad.abs()+1e-3)).max()); row['grad_pass']=bool(torch.allclose(xo.grad,xi.grad,atol=3e-2,rtol=3e-2))
          row['parameter_grads']=[{'name':n,'max_abs':float((dict(original.named_parameters())[n].grad-dict(integrated.named_parameters())[n].grad).abs().max()),'pass':bool(torch.allclose(dict(original.named_parameters())[n].grad,dict(integrated.named_parameters())[n].grad,atol=3e-2,rtol=3e-2))} for n,p in integrated.named_parameters()]
        if mode=='train' and [S,B,H]==[128,2,1024]:
          go=torch.randn_like(x); samples=[]
          for model in (original, integrated):
            for _ in range(5):
              model.zero_grad(set_to_none=True); x0=x.detach().requires_grad_(); y,_=model(x0); y.backward(go); torch.cuda.synchronize()
            for _ in range(5):
              model.zero_grad(set_to_none=True); x0=x.detach().requires_grad_(); a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); y,_=model(x0); y.backward(go); b.record(); torch.cuda.synchronize(); samples.append(a.elapsed_time(b)*1000)
          row['train_original_samples_us']=samples[:5]; row['train_integrated_samples_us']=samples[5:]
          row['train_original_mean_us']=statistics.mean(samples[:5]); row['train_integrated_mean_us']=statistics.mean(samples[5:]); row['train_speedup']=row['train_original_mean_us']/row['train_integrated_mean_us']
        rows.append(row)
    finally: mlp_mod.bias_swiglu_impl=old
    return {'mode':mode,'commit':COMMIT,'shapes':rows,'kernel':'launch_swiglu_stream','current_stream':True}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['original','benchmark','profile','grad','train'],default='benchmark'); a=ap.parse_args(); print(json.dumps(run(a.mode),indent=2))
