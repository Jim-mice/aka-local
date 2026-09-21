import argparse, ctypes, json, math, os, subprocess, statistics, torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

SHAPES = [(16,1,1024), (64,2,1024), (128,2,1024)]
NVCC = '/usr/local/cuda-11.8/bin/nvcc'

class LocalLinear(nn.Module):
    def __init__(self, ins, outs, **kw):
        super().__init__(); self.weight = nn.Parameter(torch.randn(outs, ins, device='cuda', dtype=torch.float16) * 0.01)
    def forward(self, x): return F.linear(x, self.weight), None

def make_mlp():
    cfg=TransformerConfig(num_layers=1, hidden_size=1024, num_attention_heads=16, ffn_hidden_size=4096, gated_linear_unit=True, activation_func=F.silu, add_bias_linear=False, bias_activation_fusion=False, use_te_activation_func=False, tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    subs=MLPSubmodules(linear_fc1=lambda *a,**k: LocalLinear(a[0],a[1]), linear_fc2=lambda *a,**k: LocalLinear(a[0],a[1]))
    return MLP(cfg, subs, input_size=1024, ffn_hidden_size=4096).to(device='cuda', dtype=torch.float16).eval()

def analytical(intermediate, bias, offset):
    if bias is not None: intermediate = intermediate + bias
    gate, up = torch.chunk(intermediate, 2, dim=-1)
    return torch.nn.functional.silu(gate) * (up + offset)

def time_cuda(fn, warmup=10, iters=100):
    for _ in range(warmup): fn()
    samples=[]
    for _ in range(5):
        torch.cuda.synchronize(); a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
        for _ in range(iters): fn()
        b.record(); torch.cuda.synchronize(); samples.append(a.elapsed_time(b) * 1000 / iters)
    mean=statistics.mean(samples); std=statistics.stdev(samples) if len(samples)>1 else 0.0
    return {'mean_us': mean, 'std_us': std, 'cv': std/mean if mean else 0.0, 'samples_us': samples}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidate', required=True); ap.add_argument('--profile', action='store_true'); args=ap.parse_args()
    so='/tmp/aka_swiglu_candidate.so'; compile_cmd=[NVCC,'-Xcompiler','-fPIC','-shared','-O2','-gencode','arch=compute_70,code=sm_70','-o',so,args.candidate]
    cp=subprocess.run(compile_cmd,capture_output=True,text=True); result={'compile_pass':cp.returncode==0,'compile_error':cp.stderr[-2000:],'contract_hash':'2b05cc321bed0956','source_commit':'5be9626709af2722333bf54797c954c09edeada3','shapes':[]}
    if cp.returncode: print(json.dumps(result,indent=2)); return
    lib=ctypes.CDLL(so); lib.launch_swiglu.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_float]
    torch.manual_seed(14); mlp=make_mlp(); offset=0.0
    for S,B,H in SHAPES:
        x=torch.randn(S,B,H,device='cuda',dtype=torch.float16)
        with torch.no_grad(): intermediate,_=mlp.linear_fc1(x); ref=analytical(intermediate,None,offset); out=torch.empty_like(ref)
        def cand():
            lib.launch_swiglu(intermediate.data_ptr(),0,out.data_ptr(),S*B,2*4096,offset)
        cand(); torch.cuda.synchronize(); err=(out-ref).abs(); rel=(err/(ref.abs()+1e-3)).max()
        row={'shape':[S,B,H],'dtype':'torch.float16','input_stride':list(intermediate.stride()),'output_stride':list(out.stride()),'max_abs_error':float(err.max()),'max_rel_error':float(rel),'correctness':bool(torch.allclose(out,ref,atol=2e-3,rtol=2e-3))}
        row['baseline']=time_cuda(lambda: analytical(intermediate,None,offset)); row['candidate']=time_cuda(cand); row['baseline_latency_us']=row['baseline']['mean_us']; row['candidate_latency_us']=row['candidate']['mean_us']; row['speedup']=row['baseline_latency_us']/row['candidate_latency_us']; result['shapes'].append(row)
    result['correctness_pass']=all(x['correctness'] for x in result['shapes']); result['geometric_mean_speedup']=math.prod(x['speedup'] for x in result['shapes'])**(1/len(result['shapes']))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
