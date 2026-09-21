import ctypes, os, torch, torch.distributed as dist
from remote_comparable_bench import run_max, run_prepare, common_finalize_backward, candidate_functions

def timed(fn):
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True); a.record(); fn(); b.record(); return a, b

def main():
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); mode=os.environ.get('MODE','candidate'); torch.cuda.set_device(local); dist.init_process_group('nccl'); lib=candidate_functions(os.environ['SO'])
    rows=64; V=64; world=2; lv=V//world; torch.manual_seed(764); full=torch.randn(rows,V,device='cuda',dtype=torch.float16); target=torch.randint(0,V,(rows,),device='cuda',dtype=torch.long); x=full[:,rank*lv:(rank+1)*lv].contiguous(); lm=torch.empty(rows,device='cuda',dtype=torch.float32); pred=torch.empty_like(lm); den=torch.empty_like(lm); ex=torch.empty(rows,lv,device='cuda',dtype=torch.float32); mask=torch.empty(rows*lv,device='cuda',dtype=torch.uint8); tl=torch.empty(rows,device='cuda',dtype=torch.long); loss=torch.empty_like(lm); soft=torch.empty_like(ex); grad=torch.empty_like(ex); stream=torch.cuda.current_stream()
    for _ in range(5):
        dist.barrier(); run_max(mode,lib,x,lm,rows,lv,stream); dist.all_reduce(lm,op=dist.ReduceOp.MAX); run_prepare(mode,lib,x,target,lm,pred,den,ex,mask,tl,rows,lv,rank,stream); dist.all_reduce(pred); dist.all_reduce(den); common_finalize_backward(pred,den,ex,mask,tl,loss,soft,grad,target,rank,lv)
    torch.cuda.synchronize(); out=[]
    for iteration in range(10):
        dist.barrier(); torch.cuda.synchronize(); ev={}
        ev['total']=timed(lambda: None)
        ev['max_local']=timed(lambda: run_max(mode,lib,x,lm,rows,lv,stream)); dist.all_reduce(lm,op=dist.ReduceOp.MAX)
        ev['max_collective']=timed(lambda: None)
        ev['prepare_local']=timed(lambda: run_prepare(mode,lib,x,target,lm,pred,den,ex,mask,tl,rows,lv,rank,stream)); ev['sum_target']=timed(lambda: dist.all_reduce(pred)); ev['sum_den']=timed(lambda: dist.all_reduce(den)); ev['common']=timed(lambda: common_finalize_backward(pred,den,ex,mask,tl,loss,soft,grad,target,rank,lv)); torch.cuda.synchronize()
        out.append({'iteration':iteration,'stages_us':{k:v[0].elapsed_time(v[1])*1000.0 for k,v in ev.items()}})
    print({'rank':rank,'mode':mode,'rows':out},flush=True); dist.barrier(); dist.destroy_process_group()
if __name__=='__main__': main()
