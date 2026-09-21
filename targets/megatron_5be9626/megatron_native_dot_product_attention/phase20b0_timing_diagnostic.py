"""Non-scoring reconciliation of real native core timing domains."""
import base64,json,pathlib,paramiko
root=pathlib.Path(__file__).resolve().parents[3]; out=root/'targets/megatron_5be9626/megatron_native_dot_product_attention'; secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
remote=r'''import json,time,statistics,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29830',rank=0,world_size=1); parallel_state.initialize_model_parallel(tensor_model_parallel_size=1); model_parallel_cuda_manual_seed(2030); pg=get_default_pg_collection(); results=[]
def stats(x): return {'n':len(x),'mean_us':sum(x)/len(x),'std_us':statistics.stdev(x),'cv':statistics.stdev(x)/(sum(x)/len(x)),'min_us':min(x),'max_us':max(x)}
for S,B in [(16,1),(64,2),(128,2)]:
 cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,num_query_groups=16,kv_channels=64,attention_dropout=0.,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16); m=DotProductAttention(cfg,1,AttnMaskType.no_mask,'self',pg_collection=pg).cuda().eval(); q=torch.randn(S,B,16,64,device='cuda',dtype=torch.float16);k=torch.randn_like(q);v=torch.randn_like(q)
 def hybrid():
  q3=q.reshape(S,B*16,64);k3=k.view(S,B*16,64); sc=torch.empty((B*16,S,S),device='cuda',dtype=torch.float16);sc=torch.baddbmm(sc,q3.transpose(0,1),k3.transpose(0,1).transpose(1,2),beta=0.,alpha=.125);p=torch.softmax(sc.view(B,16,S,S).float(),dim=-1).half();c=torch.bmm(p.view(B*16,S,S),v.view(S,B*16,64).transpose(0,1));return c.view(B,16,S,64).permute(2,0,1,3).contiguous().view(S,B,1024)
 for _ in range(10):m(q,k,v,None);hybrid()
 torch.cuda.synchronize(); st=torch.cuda.Event(True); en=torch.cuda.Event(True); ev=[]; wall=[]; hywall=[]; batch=[]; empty_sync=[]
 for _ in range(30):
  st.record();m(q,k,v,None);en.record();en.synchronize();ev.append(st.elapsed_time(en)*1000)
  t=time.perf_counter_ns();m(q,k,v,None);torch.cuda.synchronize();wall.append((time.perf_counter_ns()-t)/1000)
  t=time.perf_counter_ns();hybrid();torch.cuda.synchronize();hywall.append((time.perf_counter_ns()-t)/1000)
  t=time.perf_counter_ns();torch.cuda.synchronize();empty_sync.append((time.perf_counter_ns()-t)/1000)
  st.record()
  for z in range(16):m(q,k,v,None)
  en.record();en.synchronize();batch.append(st.elapsed_time(en)*1000/16)
 results.append({'shape':[S,B,1024],'cuda_event_single':stats(ev),'wall_real_call_sync':stats(wall),'wall_hybrid_call_sync':stats(hywall),'cuda_event_batched_per_call':stats(batch),'empty_synchronize_host':stats(empty_sync)})
print(json.dumps({'status':'DIAGNOSTIC_ONLY_NOT_SCORING','events':'created once per shape and reused on current stream; event creation excluded','results':results,'collectives':0}));parallel_state.destroy_model_parallel();dist.destroy_process_group()'''
import base64; d='/tmp/aka_phase20b0_timing'; enc=base64.b64encode(remote.encode()).decode(); s.exec_command(f'mkdir -p {d}; echo {enc} | base64 -d > {d}/run.py')[1].read(); cmd=f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/run.py'; _,o,e=s.exec_command(cmd,timeout=1200); (out/'phase20b0_timing_diagnostic.json').write_text(json.dumps({'command':cmd,'stdout':o.read().decode(),'stderr':e.read().decode()},indent=2)+'\n');print('written');s.close()
