import base64, json, pathlib, paramiko, hashlib
root=pathlib.Path(__file__).resolve().parents[3]
ep=root/'campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention/episode_A2'
src=ep/'candidate.cu'; secret=__import__('os').environ['AKA_V100_PASSWORD']
ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()); ssh.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
d='/tmp/aka_phase17b_a2'
remote='''import ctypes,sys,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
lib=ctypes.CDLL("/tmp/aka_phase17b_a2/candidate.so")
f=lib.dot_product_attention_forward_fp16_stream
f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_float,ctypes.c_void_p]
S=int(sys.argv[1]); B=int(sys.argv[2]); mode=sys.argv[3]
dist.init_process_group("gloo",init_method="tcp://127.0.0.1:29617",rank=0,world_size=1); parallel_state._set_global_memory_buffer(); get_cuda_rng_tracker().add("model-parallel-rng",1234); pg=ProcessGroupCollection(); pg.tp=dist.group.WORLD
cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,num_query_groups=16,kv_channels=64,attention_dropout=0.0,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16)
m=DotProductAttention(cfg,1,AttnMaskType.no_mask,"self",pg_collection=pg).cuda(); torch.manual_seed(1717); q=torch.randn(S,B,16,64,device="cuda",dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q); out=torch.empty(S,B,1024,device="cuda",dtype=torch.float16)
def candidate(): f(q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),S,B,16,64,.125,torch.cuda.current_stream().cuda_stream)
def reference(): m(q,k,v,None)
fn=candidate if mode=="candidate" else reference
for _ in range(5): fn()
torch.cuda.synchronize(); torch.cuda.profiler.start(); torch.cuda.nvtx.range_push("AKA_ATTENTION_"+mode.upper()); fn(); torch.cuda.nvtx.range_pop(); torch.cuda.synchronize(); torch.cuda.profiler.stop(); dist.destroy_process_group()
'''
enc=base64.b64encode(remote.encode()).decode(); ssh.exec_command(f'rm -f {d}/profile.py; echo {enc} | base64 -d > {d}/attention_profile_run.py')[1].read()
records=[]
for S,B in [(16,1),(64,2),(128,2)]:
 for mode in ('reference','candidate'):
  name=f'a2_{mode}_s{S}_b{B}'; cmd=f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD /usr/local/cuda-11.3/bin/nsys profile --force-overwrite=true --trace=cuda,nvtx --capture-range=cudaProfilerApi --stop-on-exit=true -o {d}/{name} <REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/attention_profile_run.py {S} {B} {mode}'
  _,o,e=ssh.exec_command(cmd,timeout=900); out=o.read().decode(); err=e.read().decode(); _,so,se=ssh.exec_command(f'/usr/local/cuda-11.3/bin/nsys stats --report gpukernsum,cudaapisum,nvtxppsum --format json --output {d}/{name}_stats {d}/{name}.qdrep',timeout=300); records.append({'shape':[S,B,1024],'mode':mode,'name':name,'stdout':out[-4000:],'stderr':err[-4000:],'stats':so.read().decode()[-12000:],'stats_err':se.read().decode()[-4000:]})
(ep/'nsys_raw.json').write_text(json.dumps({'candidate_source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'records':records},indent=2)+'\n'); print(json.dumps({'records':len(records),'artifact':str(ep/'nsys_raw.json')},indent=2)); ssh.close()
