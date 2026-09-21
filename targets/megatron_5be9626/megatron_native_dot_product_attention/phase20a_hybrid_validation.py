"""Evaluator-validation-only vendor-GEMM-preserving attention prototype."""
import base64, json, pathlib, paramiko

root=pathlib.Path(__file__).resolve().parents[3]; out=root/'targets/megatron_5be9626/megatron_native_dot_product_attention'
secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
remote=r'''import json,time,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29820',rank=0,world_size=1); parallel_state.initialize_model_parallel(tensor_model_parallel_size=1); model_parallel_cuda_manual_seed(2020); pg=get_default_pg_collection(); ans=[]
def hybrid(q,k,v):
 S,B,N,D=q.shape; q3=q.reshape(S,B*N,D); k3=k.view(S,B*N,D); scores=torch.empty((B*N,S,S),device='cuda',dtype=q.dtype)
 scores=torch.baddbmm(scores,q3.transpose(0,1),k3.transpose(0,1).transpose(1,2),beta=0.0,alpha=0.125)
 probs=torch.softmax(scores.view(B,N,S,S).float(),dim=-1).half(); v3=v.view(S,B*N,D); ctx=torch.bmm(probs.view(B*N,S,S),v3.transpose(0,1)); return ctx.view(B,N,S,D).permute(2,0,1,3).contiguous().view(S,B,N*D)
for S,B in [(16,1),(64,2),(128,2)]:
 cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,num_query_groups=16,kv_channels=64,attention_dropout=0.0,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16)
 m=DotProductAttention(cfg,1,AttnMaskType.no_mask,'self',pg_collection=pg).cuda().eval(); q=torch.randn(S,B,16,64,device='cuda',dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q); ref=m(q,k,v,None); got=hybrid(q,k,v); oracle=torch.softmax((q.float().permute(1,2,0,3)@k.float().permute(1,2,3,0))*0.125,dim=-1)@v.float().permute(1,2,0,3); oracle=oracle.permute(2,0,1,3).reshape(S,B,1024).half(); err=(got-ref).abs(); eo=(got-oracle).abs()
 def tm(fn):
  for _ in range(5): fn()
  torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); vals=[]
  for _ in range(20): a.record(); fn(); b.record(); b.synchronize(); vals.append(a.elapsed_time(b)*1000)
  return sum(vals)/len(vals)
 q3=q.reshape(S,B*16,64); k3=k.view(S,B*16,64); score=torch.empty((B*16,S,S),device='cuda',dtype=torch.float16); p=torch.empty((B,16,S,S),device='cuda',dtype=torch.float16); v3=v.view(S,B*16,64)
 qk=lambda:torch.baddbmm(score,q3.transpose(0,1),k3.transpose(0,1).transpose(1,2),beta=0.,alpha=.125); sm=lambda:torch.softmax(score.view(B,16,S,S).float(),dim=-1).half(); pv=lambda:torch.bmm(p.view(B*16,S,S),v3.transpose(0,1))
 ans.append({'shape':[S,B,1024],'max_abs_reference':float(err.max()),'max_abs_oracle':float(eo.max()),'finite':bool(torch.isfinite(got).all()),'micro_us':{'vendor_qk':tm(qk),'fp32_softmax_cast':tm(sm),'vendor_pv':tm(pv),'hybrid_end_to_end':tm(lambda:hybrid(q,k,v))},'workspace_bytes':{'scores_fp16':B*16*S*S*2,'softmax_fp32_transient':B*16*S*S*4,'probabilities_fp16':B*16*S*S*2}})
print(json.dumps({'label':'EVALUATOR_VALIDATION_ONLY_NOT_AGENT_CANDIDATE_NOT_BENCHMARK_ELIGIBLE','results':ans,'collectives':0})); parallel_state.destroy_model_parallel(); dist.destroy_process_group()'''
d='/tmp/aka_phase20a_hybrid'; enc=base64.b64encode(remote.encode()).decode(); s.exec_command(f'mkdir -p {d}; echo {enc} | base64 -d > {d}/run.py')[1].read(); cmd=f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/run.py'; _,o,e=s.exec_command(cmd,timeout=1200); result={'command':cmd,'stdout':o.read().decode(),'stderr':e.read().decode()}; (out/'phase20a_hybrid_validation.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'artifact':str(out/'phase20a_hybrid_validation.json'),'stderr_tail':result['stderr'][-500:]},indent=2)); s.close()
