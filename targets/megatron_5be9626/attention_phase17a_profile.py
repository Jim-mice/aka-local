import base64, json, paramiko
from pathlib import Path

root = Path(__file__).resolve().parents[2]
secret = __import__("os").environ["AKA_V100_PASSWORD"]
ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()); ssh.connect("<REMOTE_HOST>", username="<REMOTE_USER>", password=secret, timeout=20)
remote = r'''import argparse,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
p=argparse.ArgumentParser(); p.add_argument('--S',type=int); p.add_argument('--B',type=int); p.add_argument('--mode',default='forward'); a=p.parse_args()
dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29596',rank=0,world_size=1); parallel_state._set_global_memory_buffer(); get_cuda_rng_tracker().add('model-parallel-rng',1234); pg=ProcessGroupCollection(); pg.tp=dist.group.WORLD
cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,num_query_groups=16,kv_channels=64,attention_dropout=0.0,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16); m=DotProductAttention(cfg,1,AttnMaskType.no_mask,'self',pg_collection=pg).cuda(); S,B,H,D=a.S,a.B,16,64; torch.manual_seed(1717); q=torch.randn(S,B,H,D,device='cuda',dtype=torch.float16,requires_grad=a.mode=='backward'); k=torch.randn_like(q,requires_grad=a.mode=='backward'); v=torch.randn_like(q,requires_grad=a.mode=='backward'); y=m(q,k,v,None); g=torch.ones_like(y); torch.cuda.synchronize(); torch.cuda.profiler.start(); torch.cuda.nvtx.range_push('AKA_MEGA_DOT_PRODUCT_ATTENTION_'+a.mode.upper()+'_BEGIN');
if a.mode=='forward': y=m(q.detach(),k.detach(),v.detach(),None)
else: y.backward(g)
torch.cuda.nvtx.range_pop(); torch.cuda.synchronize(); torch.cuda.profiler.stop(); dist.destroy_process_group()
'''
payload = base64.b64encode(remote.encode()).decode(); ssh.exec_command("echo " + payload + " | base64 -d > /tmp/aka_phase17a_profile.py")[1].read()
results=[]
for mode in ("forward", "backward"):
    outbase=f"/tmp/aka_phase17a_{mode}"
    cmd=f"cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD /usr/local/cuda-11.8/bin/nsys profile --force-overwrite=true --trace=cuda,nvtx --capture-range=cudaProfilerApi --stop-on-exit=true -o {outbase} <REMOTE_HOME>/venvs/lerobot-act/bin/python /tmp/aka_phase17a_profile.py --S 64 --B 2 --mode {mode}"
    _,o,e=ssh.exec_command(cmd,timeout=600); text=o.read().decode(); err=e.read().decode()
    _,so,se=ssh.exec_command(f"/usr/local/cuda-11.8/bin/nsys stats --force-export=true --report gpukernsum,cudaapisum,nvtxppsum --format json {outbase}.qdrep",timeout=600)
    results.append({'mode':mode,'profile_stdout':text[-4000:],'profile_stderr':err[-2000:],'stats_stdout':so.read().decode()[-12000:],'stats_stderr':se.read().decode()[-2000:]})
(root/'targets/megatron_5be9626/attention_phase17a_nsys_summary.json').write_text(json.dumps(results,indent=2)+'\n'); print(json.dumps(results,indent=2)); ssh.close()
