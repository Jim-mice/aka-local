import argparse,json,os,statistics,torch,torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

COMMIT='5be9626709af2722333bf54797c954c09edeada3'; SHAPES=[(8,1,32),(32,2,64),(128,2,128)]
def setup():
    rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1')); local=int(os.environ.get('LOCAL_RANK','0'))
    if torch.cuda.is_available(): torch.cuda.set_device(local); device=torch.device('cuda')
    else: device=torch.device('cpu')
    dist.init_process_group('nccl',rank=rank,world_size=world)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world, pipeline_model_parallel_size=1)
    return rank,world,device
def one(S,B,V,world,device,bench=False):
    N=S*B
    if V % world != 0: raise ValueError(f'global vocabulary {V} is not divisible by TP={world}')
    torch.manual_seed(1500+N+V)
    full=torch.randn(N,V,device=device,dtype=torch.float16,requires_grad=True); target=torch.randint(0,V,(N,),device=device,dtype=torch.long)
    start=int(os.environ.get('RANK','0'))*V//world; end=(int(os.environ.get('RANK','0'))+1)*V//world
    local=full.detach().clone()[:,start:end].contiguous().requires_grad_(); shard=local
    # Reconstruct the exact distributed input from the same deterministic global fixture.
    loss=vocab_parallel_cross_entropy(shard.view(S,B,-1),target.view(S,B),tp_group=dist.group.WORLD if world>1 else None)
    oracle=F.cross_entropy(full,target,reduction='none')
    oracle_cmp=oracle.to(loss.dtype); err=(loss.reshape(-1)-oracle_cmp).abs(); loss.sum().backward();
    # Full oracle gradient is computed independently; compare the local shard.
    og=torch.autograd.grad(oracle.sum(),full,retain_graph=False)[0] if False else None
    # recompute oracle gradient without disturbing the distributed graph
    full2=full.detach().clone().requires_grad_(); F.cross_entropy(full2,target,reduction='none').sum().backward(); expected=full2.grad[:,(int(os.environ.get('RANK','0'))*V//world):((int(os.environ.get('RANK','0'))+1)*V//world)]
    ge=(local.grad-expected).abs()
    row={'shape':[S,B,V],'rank':int(os.environ.get('RANK','0')),'world':world,'local_vocab_range':[start,end],'loss_max_abs':float(err.max()),'loss_max_rel':float((err/(oracle_cmp.abs()+1e-5)).max()),'grad_max_abs':float(ge.max()),'grad_max_rel':float((ge/(expected.abs()+1e-5)).max()),'pass':bool(torch.allclose(loss.reshape(-1),oracle_cmp,atol=2e-3,rtol=2e-3) and torch.allclose(local.grad,expected,atol=3e-2,rtol=3e-2))}
    if bench:
        for _ in range(10):
            z=local.detach().clone().requires_grad_(); ss=z.view(S,B,-1); y=vocab_parallel_cross_entropy(ss,target.view(S,B),tp_group=dist.group.WORLD if world>1 else None); y.sum().backward()
        if world>1: dist.barrier()
        samples=[]
        for _ in range(5):
            if world>1: dist.barrier()
            a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record(); z=local.detach().clone().requires_grad_(); ss=z.view(S,B,-1); y=vocab_parallel_cross_entropy(ss,target.view(S,B),tp_group=dist.group.WORLD if world>1 else None); y.sum().backward(); b.record(); torch.cuda.synchronize(); samples.append(a.elapsed_time(b)*1000)
        row['samples_us']=samples; row['mean_us']=statistics.mean(samples); row['std_us']=statistics.stdev(samples); row['cv']=row['std_us']/row['mean_us']
    return row
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bench',action='store_true'); a=ap.parse_args(); rank,world,device=setup(); rows=[one(*s,world,device,a.bench) for s in SHAPES]; print(json.dumps({'commit':COMMIT,'rank':rank,'world':world,'dtype':'torch.float16','rows':rows},indent=2),flush=True); 
    parallel_state.destroy_model_parallel(); dist.barrier(); dist.destroy_process_group()
if __name__=='__main__': main()
