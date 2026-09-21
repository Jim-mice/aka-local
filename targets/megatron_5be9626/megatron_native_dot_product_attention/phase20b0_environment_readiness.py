"""One read-only Phase 20-B.0 readiness assessment, not scoring."""
import base64,json,pathlib,paramiko
root=pathlib.Path(__file__).resolve().parents[3]; out=root/'targets/megatron_5be9626/megatron_native_dot_product_attention'; secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
query='nvidia-smi --query-gpu=index,uuid,pci.bus_id,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader,nounits'; _,bo,be=s.exec_command(query,timeout=30);before=bo.read().decode()
remote=r'''import json,statistics,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29832',rank=0,world_size=1);parallel_state.initialize_model_parallel(tensor_model_parallel_size=1);model_parallel_cuda_manual_seed(2032);pg=get_default_pg_collection();cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,num_query_groups=16,kv_channels=64,attention_dropout=0.,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16);m=DotProductAttention(cfg,1,AttnMaskType.no_mask,'self',pg_collection=pg).cuda().eval();q=torch.randn(64,2,16,64,device='cuda',dtype=torch.float16);k=torch.randn_like(q);v=torch.randn_like(q)
for _ in range(5):m(q,k,v,None)
torch.cuda.synchronize();a=torch.cuda.Event(True);b=torch.cuda.Event(True);x=[]
for _ in range(20):a.record();m(q,k,v,None);b.record();b.synchronize();x.append(a.elapsed_time(b)*1000)
mean=sum(x)/len(x);print(json.dumps({'kind':'REFERENCE_ONLY_ENVIRONMENT_READINESS_NOT_SCORING','samples_us':x,'mean_us':mean,'cv':statistics.stdev(x)/mean}));parallel_state.destroy_model_parallel();dist.destroy_process_group()''';d='/tmp/aka_phase20b0_env';enc=base64.b64encode(remote.encode()).decode();s.exec_command(f'mkdir -p {d}; echo {enc} | base64 -d > {d}/run.py')[1].read();_,po,pe=s.exec_command(f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/run.py',timeout=900);_,ao,ae=s.exec_command(query,timeout=30);(out/'phase20b0_environment_readiness.json').write_text(json.dumps({'status':'READINESS_DIAGNOSTIC_ONLY','before':before,'probe_stdout':po.read().decode(),'probe_stderr':pe.read().decode()[-1000:],'after':ao.read().decode()},indent=2)+'\n');print('written');s.close()
