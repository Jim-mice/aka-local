"""Exact-source stage event diagnostic; no candidate and no scoring."""
import base64,json,pathlib,paramiko
root=pathlib.Path(__file__).resolve().parents[3]; out=root/'targets/megatron_5be9626/megatron_native_dot_product_attention'; secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
remote=r'''import json,statistics,torch,torch.distributed as dist
from megatron.core import parallel_state,tensor_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29831',rank=0,world_size=1); parallel_state.initialize_model_parallel(tensor_model_parallel_size=1); model_parallel_cuda_manual_seed(2031); pg=get_default_pg_collection(); results=[]
def mean(x):return sum(x)/len(x)
for S,B in [(16,1),(64,2),(128,2)]:
 cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,num_query_groups=16,kv_channels=64,attention_dropout=0.,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16);m=DotProductAttention(cfg,1,AttnMaskType.no_mask,'self',pg_collection=pg).cuda().eval();q=torch.randn(S,B,16,64,device='cuda',dtype=torch.float16);k=torch.randn_like(q);v=torch.randn_like(q)
 def manual():
  os=(q.size(1),q.size(2),q.size(0),k.size(0));qq=q.reshape(os[2],os[0]*os[1],-1);kk=k.view(os[3],os[0]*os[1],-1);buf=parallel_state.get_global_memory_buffer().get_tensor((os[0]*os[1],os[2],os[3]),q.dtype,'mpu');r=torch.baddbmm(buf,qq.transpose(0,1),kk.transpose(0,1).transpose(1,2),beta=0.,alpha=m.softmax_scale);p=m.scale_mask_softmax(r.view(*os),None,m.softmax_offset)
  with tensor_parallel.get_cuda_rng_tracker().fork(): p=m.attention_dropout(p)
  os2=(v.size(1),v.size(2),qq.size(0),v.size(3));vv=v.view(v.size(0),os2[0]*os2[1],-1);c=torch.bmm(p.view(os2[0]*os2[1],os2[2],-1),vv.transpose(0,1));return c.view(*os2).permute(2,0,1,3).contiguous().view(S,B,1024)
 for _ in range(8):m(q,k,v,None);manual()
 torch.cuda.synchronize(); st=[torch.cuda.Event(True) for _ in range(7)]; vals=[[] for _ in range(6)]; total=[]; diff=[]
 for _ in range(20):
  os=(q.size(1),q.size(2),q.size(0),k.size(0));qq=q.reshape(os[2],os[0]*os[1],-1);kk=k.view(os[3],os[0]*os[1],-1); st[0].record();buf=parallel_state.get_global_memory_buffer().get_tensor((os[0]*os[1],os[2],os[3]),q.dtype,'mpu');st[1].record();r=torch.baddbmm(buf,qq.transpose(0,1),kk.transpose(0,1).transpose(1,2),beta=0.,alpha=m.softmax_scale);st[2].record();p=m.scale_mask_softmax(r.view(*os),None,m.softmax_offset);st[3].record()
  with tensor_parallel.get_cuda_rng_tracker().fork():p=m.attention_dropout(p)
  st[4].record();os2=(v.size(1),v.size(2),qq.size(0),v.size(3));vv=v.view(v.size(0),os2[0]*os2[1],-1);c=torch.bmm(p.view(os2[0]*os2[1],os2[2],-1),vv.transpose(0,1));st[5].record();z=c.view(*os2).permute(2,0,1,3).contiguous().view(S,B,1024);st[6].record();st[6].synchronize();
  for i in range(6):vals[i].append(st[i].elapsed_time(st[i+1])*1000)
  total.append(st[0].elapsed_time(st[6])*1000); a=m(q,k,v,None);torch.cuda.synchronize();diff.append(float((z-a).abs().max()))
 results.append({'shape':[S,B,1024],'stages_us':{'global_buffer_lookup_to_event':mean(vals[0]),'vendor_qk':mean(vals[1]),'softmax_conversion':mean(vals[2]),'dropout_p0_tracker':mean(vals[3]),'vendor_pv':mean(vals[4]),'output_permute_contiguous':mean(vals[5])},'manual_total_event_us':mean(total),'manual_vs_real_max_abs':max(diff)})
print(json.dumps({'status':'DIAGNOSTIC_ONLY','results':results,'collectives':0}));parallel_state.destroy_model_parallel();dist.destroy_process_group()'''
d='/tmp/aka_phase20b0_stage';enc=base64.b64encode(remote.encode()).decode();s.exec_command(f'mkdir -p {d}; echo {enc} | base64 -d > {d}/run.py')[1].read();cmd=f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/run.py';_,o,e=s.exec_command(cmd,timeout=1200);(out/'phase20b0_stage_reconciliation.json').write_text(json.dumps({'command':cmd,'stdout':o.read().decode(),'stderr':e.read().decode()},indent=2)+'\n');print('written');s.close()
