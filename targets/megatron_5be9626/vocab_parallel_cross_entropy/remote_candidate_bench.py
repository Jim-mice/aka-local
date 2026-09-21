import ctypes,os,statistics,torch,torch.distributed as dist
from megatron.core import parallel_state
def main():
 r=int(os.environ['RANK']); w=int(os.environ['WORLD_SIZE']); lr=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr); dist.init_process_group('nccl'); parallel_state.initialize_model_parallel(tensor_model_parallel_size=w,pipeline_model_parallel_size=1)
 lib=ctypes.CDLL(os.environ['SO']); lib.local_max_fp16_stream.restype=ctypes.c_int; lib.local_prepare_fp16_stream.restype=ctypes.c_int
 rows_list=[8,64,256]; v_list=[32,64,128]; out=[]
 for rows,V in zip(rows_list,v_list):
  lv=V//w; torch.manual_seed(700+rows); full=torch.randn(rows,V,device='cuda',dtype=torch.float16); target=torch.randint(0,V,(rows,),device='cuda'); x=full[:,r*lv:(r+1)*lv].contiguous(); times=[]
  for _ in range(5):
   lm=torch.empty(rows,device='cuda',dtype=torch.float32); pred=torch.empty_like(lm); den=torch.empty_like(lm); ex=torch.empty(rows,lv,device='cuda',dtype=torch.float32); mask=torch.empty(rows*lv,device='cuda',dtype=torch.uint8); tl=torch.empty(rows,device='cuda',dtype=torch.long); s=torch.cuda.current_stream(); dist.barrier(); lib.local_max_fp16_stream(ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(lm.data_ptr()),rows,lv,ctypes.c_void_p(s.cuda_stream)); dist.all_reduce(lm,op=dist.ReduceOp.MAX); lib.local_prepare_fp16_stream(ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(target.data_ptr()),ctypes.c_void_p(lm.data_ptr()),ctypes.c_void_p(pred.data_ptr()),ctypes.c_void_p(den.data_ptr()),ctypes.c_void_p(ex.data_ptr()),ctypes.c_void_p(mask.data_ptr()),ctypes.c_void_p(tl.data_ptr()),rows,lv,r,ctypes.c_void_p(s.cuda_stream)); dist.all_reduce(pred); dist.all_reduce(den)
  for mi in range(6):
   lm=torch.empty(rows,device='cuda',dtype=torch.float32); pred=torch.empty_like(lm); den=torch.empty_like(lm); ex=torch.empty(rows,lv,device='cuda',dtype=torch.float32); mask=torch.empty(rows*lv,device='cuda',dtype=torch.uint8); tl=torch.empty(rows,device='cuda',dtype=torch.long); s=torch.cuda.current_stream(); dist.barrier(); torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record(); lib.local_max_fp16_stream(ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(lm.data_ptr()),rows,lv,ctypes.c_void_p(s.cuda_stream)); dist.all_reduce(lm,op=dist.ReduceOp.MAX); lib.local_prepare_fp16_stream(ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(target.data_ptr()),ctypes.c_void_p(lm.data_ptr()),ctypes.c_void_p(pred.data_ptr()),ctypes.c_void_p(den.data_ptr()),ctypes.c_void_p(ex.data_ptr()),ctypes.c_void_p(mask.data_ptr()),ctypes.c_void_p(tl.data_ptr()),rows,lv,r,ctypes.c_void_p(s.cuda_stream)); dist.all_reduce(pred); dist.all_reduce(den); soft=ex/den[:,None]; idx=(target-r*lv).clamp(0,lv-1); owned=(target>=r*lv)&(target<(r+1)*lv); soft[torch.arange(rows,device='cuda'),idx]-=owned.float(); soft.mul_(torch.ones(rows,device='cuda')[:,None]); b.record();
   torch.cuda.synchronize()
   if mi>0: times.append(a.elapsed_time(b)*1000)
  out.append({'rows':rows,'V':V,'mean_us':statistics.mean(times),'std_us':statistics.stdev(times),'cv':statistics.stdev(times)/statistics.mean(times),'samples_us':times})
 print({'rank':r,'world':w,'rows':out},flush=True); parallel_state.destroy_model_parallel(); dist.barrier(); dist.destroy_process_group()
if __name__=='__main__': main()
