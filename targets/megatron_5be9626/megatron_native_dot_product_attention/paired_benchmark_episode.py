import base64,json,paramiko
from pathlib import Path
root=Path(__file__).resolve().parents[3]; epname='A2'; ep=root/'campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention'/f'episode_{epname}'; src=ep/'candidate.cu'; secret=__import__('os').environ['AKA_V100_PASSWORD']; ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()); ssh.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20); d='/tmp/aka_phase17b_a2'; enc=base64.b64encode(src.read_bytes()).decode(); ssh.exec_command('echo '+enc+' | base64 -d > '+d+'/candidate.cu')[1].read(); ssh.exec_command('/usr/local/cuda-11.8/bin/nvcc -arch=sm_70 -O3 -shared -Xcompiler -fPIC '+d+'/candidate.cu -o '+d+'/candidate.so')[1].read()
remote=r'''import ctypes,json,statistics,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
lib=ctypes.CDLL('LIBPATH'); f=lib.dot_product_attention_forward_fp16_stream; f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_float,ctypes.c_void_p]
def cand(q,k,v,out): f(q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),q.shape[0],q.shape[1],q.shape[2],q.shape[3],ctypes.c_float(.125),torch.cuda.current_stream().cuda_stream)
def timed(fn): st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record(); fn(); en.record(); en.synchronize(); return st.elapsed_time(en)*1000.0
dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29599',rank=0,world_size=1); parallel_state._set_global_memory_buffer(); get_cuda_rng_tracker().add('model-parallel-rng',1234); pg=ProcessGroupCollection(); pg.tp=dist.group.WORLD; cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,num_query_groups=16,kv_channels=64,attention_dropout=0.0,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16); m=DotProductAttention(cfg,1,AttnMaskType.no_mask,'self',pg_collection=pg).cuda(); rows=[]
for S,B in [(16,1),(64,2),(128,2)]:
 torch.manual_seed(1717+S+B); q=torch.randn(S,B,16,64,device='cuda',dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q); out=torch.empty(S,B,1024,device='cuda',dtype=torch.float16)
 for _ in range(5): m(q,k,v,None); cand(q,k,v,out)
 torch.cuda.synchronize()
 for block in range(3):
  for iteration in range(5):
   for impl in ['reference','candidate','candidate','reference']:
    x=timed((lambda: m(q,k,v,None)) if impl=='reference' else (lambda: cand(q,k,v,out))); rows.append({'S':S,'B':B,'block':block,'iteration':iteration,'impl':impl,'latency_us':x})
print(json.dumps({'rows':rows,'collectives':0,'candidate':'A2','order':'ABBA'})); dist.destroy_process_group()
'''.replace('LIBPATH',d+'/candidate.so')
payload=base64.b64encode(remote.encode()).decode(); ssh.exec_command('echo '+payload+' | base64 -d > '+d+'/paired.py')[1].read(); runs=[]
for i in range(1,4):
    _,pre,_=ssh.exec_command("nvidia-smi --query-gpu=index,pci.bus_id,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,utilization.gpu,memory.used --format=csv,noheader,nounits",timeout=60); _,o,e=ssh.exec_command('cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python '+d+'/paired.py',timeout=900); _,post,_=ssh.exec_command("nvidia-smi --query-gpu=index,pci.bus_id,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,utilization.gpu,memory.used --format=csv,noheader,nounits",timeout=60); runs.append({'invocation':i,'pre_gpu':pre.read().decode(),'post_gpu':post.read().decode(),'stdout':o.read().decode(),'stderr':e.read().decode()[-3000:]})
(ep/'paired_raw_v2.json').write_text(json.dumps(runs,indent=2)+'\n'); print(json.dumps(runs,indent=2)); ssh.close()
