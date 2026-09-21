"""Diagnostic-only S64 NSYS: real reference versus evaluator-only hybrid."""
import base64,json,pathlib,paramiko
root=pathlib.Path(__file__).resolve().parents[3]; out=root/'targets/megatron_5be9626/megatron_native_dot_product_attention'; secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
remote=r'''import sys,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
mode=sys.argv[1]; S,B,N,D=64,2,16,64; dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29821',rank=0,world_size=1); parallel_state.initialize_model_parallel(tensor_model_parallel_size=1); model_parallel_cuda_manual_seed(2021); pg=get_default_pg_collection(); cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=N,num_query_groups=N,kv_channels=D,attention_dropout=0.0,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16); m=DotProductAttention(cfg,1,AttnMaskType.no_mask,'self',pg_collection=pg).cuda().eval(); q=torch.randn(S,B,N,D,device='cuda',dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
def hybrid():
 q3=q.reshape(S,B*N,D); k3=k.view(S,B*N,D); scores=torch.empty((B*N,S,S),device='cuda',dtype=q.dtype); scores=torch.baddbmm(scores,q3.transpose(0,1),k3.transpose(0,1).transpose(1,2),beta=0.,alpha=.125); p=torch.softmax(scores.view(B,N,S,S).float(),dim=-1).half(); c=torch.bmm(p.view(B*N,S,S),v.view(S,B*N,D).transpose(0,1)); return c.view(B,N,S,D).permute(2,0,1,3).contiguous().view(S,B,1024)
fn=(lambda:m(q,k,v,None)) if mode=='reference' else hybrid
for _ in range(8): fn()
torch.cuda.synchronize(); torch.cuda.profiler.start(); torch.cuda.nvtx.range_push('AKA_PHASE20A_HYBRID_'+mode.upper()); fn(); torch.cuda.nvtx.range_pop(); torch.cuda.synchronize(); torch.cuda.profiler.stop(); parallel_state.destroy_model_parallel(); dist.destroy_process_group()'''
d='/tmp/aka_phase20a_hybrid_nsys'; enc=base64.b64encode(remote.encode()).decode(); s.exec_command(f'mkdir -p {d}; echo {enc} | base64 -d > {d}/run.py')[1].read(); records=[]
for mode in ('reference','hybrid'):
 name=f's64_{mode}'; cmd=f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD /usr/local/cuda-11.3/bin/nsys profile --force-overwrite=true --trace=cuda,nvtx --capture-range=cudaProfilerApi --stop-on-exit=true -o {d}/{name} <REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/run.py {mode}'; _,po,pe=s.exec_command(cmd,timeout=900); _,so,se=s.exec_command(f'/usr/local/cuda-11.3/bin/nsys stats --report gpukernsum,cudaapisum,nvtxppsum --format csv --output {d}/{name}_stats {d}/{name}.qdrep',timeout=300); _,fo,fe=s.exec_command(f'for f in {d}/{name}_stats*; do echo __FILE__$f; cat $f; done',timeout=300); records.append({'mode':mode,'profile_stderr':pe.read().decode()[-2000:],'stats_stderr':se.read().decode()[-1000:],'files':fo.read().decode()[-16000:]})
(out/'phase20a_hybrid_nsys.json').write_text(json.dumps({'status':'DIAGNOSTIC_ONLY_NOT_PROMOTION','shape':[64,2,1024],'collectives':0,'records':records},indent=2)+'\n'); print(json.dumps({'artifact':str(out/'phase20a_hybrid_nsys.json')},indent=2)); s.close()
