"""Diagnostic-only M4 NSYS capture; never writes promotion evidence."""
import base64, hashlib, json, pathlib, paramiko

root=pathlib.Path(__file__).resolve().parents[3]
ep=root/'campaigns/targets/megatron_5be9626/moe_native_sequential_expert_compute/episode_M4'
src=ep/'moe_sequential_expert_forward.cu'; secret=__import__('os').environ['AKA_V100_PASSWORD']
s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
d='/tmp/aka_phase19a_m4_nsys'; s.exec_command(f'mkdir -p {d}')[1].read(); enc=base64.b64encode(src.read_bytes()).decode(); s.exec_command(f'echo {enc} | base64 -d > {d}/candidate.cu')[1].read()
_,o,e=s.exec_command(f'/usr/local/cuda-11.8/bin/nvcc -arch=sm_70 -O3 -shared -Xcompiler -fPIC {d}/candidate.cu -o {d}/candidate.so',timeout=600); compile_err=e.read().decode()
remote=r'''import ctypes,sys,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tensor_parallel.layers import ColumnParallelLinear,RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
mode=sys.argv[1]; counts=list(map(int,sys.argv[2].split(','))); E,H,I,T=4,64,128,16
dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29719',rank=0,world_size=1); parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,expert_model_parallel_size=1); model_parallel_cuda_manual_seed(1919); pg=get_default_pg_collection()
cfg=TransformerConfig(num_layers=1,hidden_size=H,num_attention_heads=4,num_moe_experts=E,moe_ffn_hidden_size=I,use_cpu_initialization=False,activation_func=torch.nn.functional.silu,gated_linear_unit=True,bias_activation_fusion=True,moe_router_topk=1,moe_router_pre_softmax=True,add_bias_linear=False,params_dtype=torch.float16,fp16=True)
m=SequentialMLP(E,cfg,MLPSubmodules(linear_fc1=ColumnParallelLinear,linear_fc2=RowParallelLinear),pg_collection=pg).cuda().eval(); x=torch.randn(T,H,device='cuda',dtype=torch.float16); p=torch.ones(T,device='cuda',dtype=torch.float16); c=torch.tensor(counts,device='cuda',dtype=torch.int64); out=torch.empty_like(x)
if mode=='candidate':
 lib=ctypes.CDLL('/tmp/aka_phase19a_m4_nsys/candidate.so'); f=lib.moe_sequential_expert_forward_fp16_stream; f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
 w1=torch.stack([z.linear_fc1.weight for z in m.local_experts]); w2=torch.stack([z.linear_fc2.weight for z in m.local_experts])
 def fn(): f(x.data_ptr(),c.data_ptr(),p.data_ptr(),w1.data_ptr(),w2.data_ptr(),out.data_ptr(),T,H,I,E,torch.cuda.current_stream().cuda_stream)
else:
 def fn(): m(x,c,p)
for _ in range(8): fn()
torch.cuda.synchronize(); torch.cuda.profiler.start(); torch.cuda.nvtx.range_push('AKA_MOE_M4_DIAGNOSTIC_'+mode.upper()+'_'+sys.argv[2]); fn(); torch.cuda.nvtx.range_pop(); torch.cuda.synchronize(); torch.cuda.profiler.stop(); parallel_state.destroy_model_parallel(); dist.destroy_process_group()'''
enc2=base64.b64encode(remote.encode()).decode(); s.exec_command(f'echo {enc2} | base64 -d > {d}/run.py')[1].read(); records=[]
for distn in ('4,4,4,4','1,3,5,7','0,0,8,8'):
 for mode in ('reference','candidate'):
  name=f'{mode}_{distn.replace(",","_")}'; cmd=f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD /usr/local/cuda-11.3/bin/nsys profile --force-overwrite=true --trace=cuda,nvtx --capture-range=cudaProfilerApi --stop-on-exit=true -o {d}/{name} <REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/run.py {mode} {distn}'
  _,po,pe=s.exec_command(cmd,timeout=900); _,so,se=s.exec_command(f'/usr/local/cuda-11.3/bin/nsys stats --report gpukernsum,cudaapisum,nvtxppsum --format csv --output {d}/{name}_stats {d}/{name}.qdrep',timeout=300); _,fo,fe=s.exec_command(f'for f in {d}/{name}_stats*; do echo __FILE__$f; cat $f; done',timeout=300)
  records.append({'distribution':distn,'mode':mode,'command':cmd,'profile_stdout':po.read().decode()[-2000:],'profile_stderr':pe.read().decode()[-3000:],'stats_stdout':so.read().decode()[-1000:],'stats_stderr':se.read().decode()[-1000:],'files':fo.read().decode()[-20000:]})
out={'status':'DIAGNOSTIC_ONLY_NOT_PROMOTION','candidate_source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'compile_stderr':compile_err,'records':records}; (ep/'diagnostic_nsys_raw.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'records':len(records),'artifact':str(ep/'diagnostic_nsys_raw.json')},indent=2)); s.close()
