import argparse, ctypes, json, os, statistics
from pathlib import Path
import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy, VocabParallelCrossEntropy

CONTRACT = "6121f49401f3ef4601549c8732a62870ff17a7a7d7047e9c55d55b79c9ff6e92"
COMMIT = "5be9626709af2722333bf54797c954c09edeada3"

def setup():
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local); dist.init_process_group('nccl'); return rank
def fixture(s,b,v,seed,rank,world):
    torch.manual_seed(seed); full=torch.randn((s,b,v),device='cuda',dtype=torch.float16); target=torch.randint(0,v,(s,b),device='cuda',dtype=torch.long); l=v//world; return full,full[...,rank*l:(rank+1)*l].contiguous(),target
def saved(local,target):
    x=local.detach().clone().requires_grad_(True); loss=vocab_parallel_cross_entropy(x,target,label_smoothing=0.0,tp_group=dist.group.WORLD); return [z.detach().clone() for z in loss.grad_fn.saved_tensors]
def ref(out,mask,idx,g):
    a,r,u,o=VocabParallelCrossEntropy.prepare_gradient_calculation_operands(out,mask); return VocabParallelCrossEntropy.calculate_gradients(a,r,idx,u,o,g)
def load(path):
    lib=ctypes.CDLL(path); f=lib.ce_backward_local_fp32_stream; f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]; f.restype=None; return f
def invoke(f,soft,mask,idx,g,out):
    f(soft.data_ptr(),mask.data_ptr(),idx.data_ptr(),g.data_ptr(),out.data_ptr(),soft.shape[0]*soft.shape[1],soft.shape[2],torch.cuda.current_stream().cuda_stream)
def err(a,b):
    d=(a-b).abs(); return {'max_abs':float(d.max()),'max_rel':float((d/b.abs().clamp_min(1e-12)).max())}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--lib'); p.add_argument('--implementation',choices=['candidate','reference'],default='candidate');p.add_argument('--S',type=int);p.add_argument('--B',type=int);p.add_argument('--V',type=int);p.add_argument('--mode',choices=['correctness','benchmark'],default='correctness');p.add_argument('--warmup',type=int,default=5);p.add_argument('--blocks',type=int,default=3);p.add_argument('--measurements',type=int,default=10);p.add_argument('--profile-gate',action='store_true');p.add_argument('--output');a=p.parse_args();rank=setup();world=dist.get_world_size();full,local,target=fixture(a.S,a.B,a.V,1515,rank,world);soft,mask,idx=saved(local,target);g=torch.linspace(.25,1.25,a.S*a.B,device='cuda',dtype=torch.float32).reshape(a.S,a.B);f=load(a.lib) if a.implementation=='candidate' else None; out=torch.empty_like(soft); before=soft.clone()
 if a.mode=='correctness':
  refout=torch.empty_like(soft); refout.copy_(soft);ref(refout,mask,idx,g);invoke(f,soft,mask,idx,g,out);torch.cuda.synchronize();result={'rank':rank,'world':world,'config':{'S':a.S,'B':a.B,'V':a.V},'contract_hash':CONTRACT,'candidate_error':err(out,refout),'softmax_readonly_error':err(soft,before),'collectives_in_boundary':0,'alias_policy':'separate_output'}
 else:
  for _ in range(a.warmup):
   if a.implementation=='candidate': invoke(f,soft,mask,idx,g,out)
   else: out.copy_(soft); ref(out,mask,idx,g)
  torch.cuda.synchronize();
  if a.profile_gate: torch.cuda.cudart().cudaProfilerStart()
  rows=[]
  for block in range(a.blocks):
   for it in range(a.measurements):
    torch.cuda.synchronize();st=torch.cuda.Event(True);en=torch.cuda.Event(True);st.record();
    if a.implementation=='candidate': invoke(f,soft,mask,idx,g,out)
    else: out.copy_(soft); ref(out,mask,idx,g)
    en.record();en.synchronize();rows.append({'rank':rank,'block':block,'iteration':it,'latency_us':st.elapsed_time(en)*1000.,'success':True,'collectives_in_boundary':0,'implementation':a.implementation})
  if a.profile_gate: torch.cuda.cudart().cudaProfilerStop()
  result={'rank':rank,'world':world,'config':{'S':a.S,'B':a.B,'V':a.V},'rows':rows,'summary':{'N':len(rows),'mean_us':statistics.mean(x['latency_us'] for x in rows),'std_us':statistics.stdev(x['latency_us'] for x in rows)},'collectives_in_boundary':0}
 print(json.dumps(result),flush=True)
 if a.output: Path(a.output.replace('{rank}',str(rank))).write_text(json.dumps(result)+'\n')
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
