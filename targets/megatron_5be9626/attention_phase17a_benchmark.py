import base64, json, paramiko, statistics
from pathlib import Path

root = Path(__file__).resolve().parents[2]
secret = __import__("os").environ["AKA_V100_PASSWORD"]
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("<REMOTE_HOST>", username="<REMOTE_USER>", password=secret, timeout=20)

remote = r'''import argparse,json,statistics,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
p=argparse.ArgumentParser(); p.add_argument('--S',type=int); p.add_argument('--B',type=int); p.add_argument('--mode',default='forward'); p.add_argument('--n',type=int,default=20); p.add_argument('--warmup',type=int,default=5); a=p.parse_args()
dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29595',rank=0,world_size=1)
parallel_state._set_global_memory_buffer(); get_cuda_rng_tracker().add('model-parallel-rng',1234)
pg=ProcessGroupCollection(); pg.tp=dist.group.WORLD
cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,num_query_groups=16,kv_channels=64,attention_dropout=0.0,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16)
m=DotProductAttention(cfg,1,AttnMaskType.no_mask,'self',pg_collection=pg).cuda(); S,B,H,D=a.S,a.B,16,64
torch.manual_seed(1717); q0=torch.randn(S,B,H,D,device='cuda',dtype=torch.float16); k0=torch.randn_like(q0); v0=torch.randn_like(q0)
def forward_once(): return m(q0,k0,v0,None)
for _ in range(a.warmup): forward_once()
torch.cuda.synchronize(); rows=[]
if a.mode=='forward':
  for _ in range(a.n):
    st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record(); forward_once(); en.record(); en.synchronize(); rows.append(st.elapsed_time(en)*1000.0)
else:
  for _ in range(a.n):
    q=q0.detach().requires_grad_(); k=k0.detach().requires_grad_(); v=v0.detach().requires_grad_(); y=m(q,k,v,None); g=torch.ones_like(y)
    torch.cuda.synchronize(); st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record(); y.backward(g); en.record(); en.synchronize(); rows.append(st.elapsed_time(en)*1000.0)
out={'shape':[S,B,1024],'mode':a.mode,'n':len(rows),'warmup':a.warmup,'samples_us':rows,'mean_us':statistics.mean(rows),'std_us':statistics.stdev(rows) if len(rows)>1 else 0.0,'min_us':min(rows),'max_us':max(rows),'collectives':0,'module':type(m).__name__}
print(json.dumps(out)); dist.destroy_process_group()
'''
payload = base64.b64encode(remote.encode()).decode()
ssh.exec_command("echo " + payload + " | base64 -d > /tmp/aka_phase17a_benchmark.py")[1].read()
results = []
for S, B in [(16, 1), (64, 2), (128, 2)]:
    for mode in ("forward", "backward"):
        cmd = f"cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python /tmp/aka_phase17a_benchmark.py --S {S} --B {B} --mode {mode} --n 20 --warmup 5"
        _, out, err = ssh.exec_command(cmd, timeout=600)
        results.append({"shape": [S, B, 1024], "mode": mode, "stdout": out.read().decode(), "stderr": err.read().decode()[-2000:]})
(root / "targets/megatron_5be9626/attention_phase17a_benchmark_raw.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
ssh.close()
