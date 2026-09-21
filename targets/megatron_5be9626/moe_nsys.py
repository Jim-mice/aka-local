import base64,json,pathlib,paramiko
root=pathlib.Path(__file__).resolve().parents[2]; out=root/'moe_native_sequential_expert_compute'; secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
remote='''import sys,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tensor_parallel.layers import ColumnParallelLinear,RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
dist.init_process_group("gloo",init_method="tcp://127.0.0.1:29620",rank=0,world_size=1); parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,expert_model_parallel_size=1); model_parallel_cuda_manual_seed(123); pg=get_default_pg_collection(); E,H,I,T=4,64,128,16; counts=list(map(int,sys.argv[1].split(',')))
cfg=TransformerConfig(num_layers=1,hidden_size=H,num_attention_heads=4,num_moe_experts=E,moe_ffn_hidden_size=I,use_cpu_initialization=False,activation_func=torch.nn.functional.silu,gated_linear_unit=True,bias_activation_fusion=True,moe_router_topk=1,moe_router_pre_softmax=True,add_bias_linear=False,params_dtype=torch.float16,fp16=True); m=SequentialMLP(E,cfg,MLPSubmodules(linear_fc1=ColumnParallelLinear,linear_fc2=RowParallelLinear),pg_collection=pg).cuda().eval(); x=torch.randn(T,H,device="cuda",dtype=torch.float16); probs=torch.ones(T,device="cuda",dtype=torch.float16); tcp=torch.tensor(counts,device="cuda",dtype=torch.int64)
for _ in range(10): m(x,tcp,probs)
torch.cuda.synchronize(); torch.cuda.profiler.start(); torch.cuda.nvtx.range_push("AKA_MOE_SEQUENTIAL_"+sys.argv[1]); m(x,tcp,probs); torch.cuda.nvtx.range_pop(); torch.cuda.synchronize(); torch.cuda.profiler.stop(); parallel_state.destroy_model_parallel(); dist.destroy_process_group()
'''; enc=base64.b64encode(remote.encode()).decode(); d='/tmp/aka_phase18a'; s.exec_command(f'mkdir -p {d}; echo {enc} | base64 -d > {d}/moe_nsys_run.py')[1].read(); rec=[]
for distn in ('4,4,4,4','1,3,5,7','0,0,8,8'):
 name='moe_'+distn.replace(',','_'); cmd=f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD /usr/local/cuda-11.3/bin/nsys profile --force-overwrite=true --trace=cuda,nvtx --capture-range=cudaProfilerApi --stop-on-exit=true -o {d}/{name} <REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/moe_nsys_run.py {distn}'; _,o,e=s.exec_command(cmd,timeout=900); _,so,se=s.exec_command(f'/usr/local/cuda-11.3/bin/nsys stats --report gpukernsum,cudaapisum,nvtxppsum --format json --output {d}/{name}_stats {d}/{name}.qdrep',timeout=300); _,fo,fe=s.exec_command(f'for f in {d}/{name}_stats*; do echo __FILE__$f; cat $f; done',timeout=300); rec.append({'distribution':distn,'command':cmd,'profile_stderr':e.read().decode()[-4000:],'stats':so.read().decode()[-1000:],'files':fo.read().decode()})
(out/'nsys_raw.json').write_text(json.dumps({'records':rec},indent=2)+'\n'); print(json.dumps({'records':len(rec),'artifact':str(out/'nsys_raw.json')},indent=2)); s.close()
