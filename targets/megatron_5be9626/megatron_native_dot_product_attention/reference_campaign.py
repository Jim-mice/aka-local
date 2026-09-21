import base64, json, paramiko, statistics, time
from pathlib import Path

root=Path(__file__).resolve().parents[3]
secret=__import__('os').environ['AKA_V100_PASSWORD']
ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()); ssh.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
remote=r'''import json,statistics,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29597',rank=0,world_size=1); parallel_state._set_global_memory_buffer(); get_cuda_rng_tracker().add('model-parallel-rng',1234); pg=ProcessGroupCollection(); pg.tp=dist.group.WORLD
cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,num_query_groups=16,kv_channels=64,attention_dropout=0.0,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16); m=DotProductAttention(cfg,1,AttnMaskType.no_mask,'self',pg_collection=pg).cuda()
allrows=[]
for S,B in [(16,1),(64,2),(128,2)]:
 torch.manual_seed(1717+S+B); q=torch.randn(S,B,16,64,device='cuda',dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
 for _ in range(5): m(q,k,v,None)
 torch.cuda.synchronize()
 for block in range(3):
  for iteration in range(10):
   st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record(); m(q,k,v,None); en.record(); en.synchronize(); allrows.append({'S':S,'B':B,'block':block,'iteration':iteration,'latency_us':st.elapsed_time(en)*1000.0})
print(json.dumps({'rows':allrows,'collectives':0,'module':'DotProductAttention'})); dist.destroy_process_group()
'''
payload=base64.b64encode(remote.encode()).decode(); ssh.exec_command("echo "+payload+" | base64 -d > /tmp/aka_phase17a_reference_campaign.py")[1].read()
runs=[]
for invocation in range(1,4):
    _,pre,preerr=ssh.exec_command("nvidia-smi --query-gpu=index,pci.bus_id,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,utilization.gpu,memory.used --format=csv,noheader,nounits",timeout=60)
    _,out,err=ssh.exec_command("cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python /tmp/aka_phase17a_reference_campaign.py",timeout=900)
    _,post,posterr=ssh.exec_command("nvidia-smi --query-gpu=index,pci.bus_id,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,utilization.gpu,memory.used --format=csv,noheader,nounits",timeout=60)
    runs.append({'invocation':invocation,'pre_gpu':pre.read().decode(),'post_gpu':post.read().decode(),'stdout':out.read().decode(),'stderr':err.read().decode()[-3000:]})
(root/'targets/megatron_5be9626/megatron_native_dot_product_attention/reference_raw.json').write_text(json.dumps(runs,indent=2)+'\n'); print(json.dumps(runs,indent=2)); ssh.close()
