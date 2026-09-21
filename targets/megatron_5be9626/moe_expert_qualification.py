import base64,json,pathlib,paramiko
root=pathlib.Path(__file__).resolve().parents[2]; out=root/'moe_native_sequential_expert_compute'; secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
remote='''import json,time,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tensor_parallel.layers import ColumnParallelLinear,RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
dist.init_process_group("gloo",init_method="tcp://127.0.0.1:29619",rank=0,world_size=1); parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,expert_model_parallel_size=1); model_parallel_cuda_manual_seed(123); pg=get_default_pg_collection(); E,H,I,T=4,64,128,16
cfg=TransformerConfig(num_layers=1,hidden_size=H,num_attention_heads=4,num_moe_experts=E,moe_ffn_hidden_size=I,use_cpu_initialization=False,activation_func=torch.nn.functional.silu,gated_linear_unit=True,bias_activation_fusion=True,moe_router_topk=1,moe_router_pre_softmax=True,add_bias_linear=False,params_dtype=torch.float16,fp16=True)
m=SequentialMLP(E,cfg,MLPSubmodules(linear_fc1=ColumnParallelLinear,linear_fc2=RowParallelLinear),pg_collection=pg).cuda().eval(); dists=[[4,4,4,4],[1,3,5,7],[0,0,8,8]]; results=[]
for counts in dists:
 x=torch.randn(T,H,device="cuda",dtype=torch.float16,requires_grad=True); probs=torch.ones(T,device="cuda",dtype=torch.float16); tcp=torch.tensor(counts,device="cuda",dtype=torch.int64); y,_=m(x,tcp,probs); yf=y.float(); oracle=torch.empty_like(yf); pos=0
 for e,n in enumerate(counts):
  if n:
   z=x[pos:pos+n].float(); w1=m.local_experts[e].linear_fc1.weight.float(); w2=m.local_experts[e].linear_fc2.weight.float(); a=z@w1.t(); a=torch.nn.functional.silu(a[:,:I])*a[:,I:]; oracle[pos:pos+n]=a@w2.t()
  pos+=n
 err=(yf-oracle).abs(); loss=y.float().sum(); loss.backward(); results.append({'distribution':counts,'max_abs':float(err.max()),'max_rel':float((err/oracle.abs().clamp_min(1e-3)).max()),'finite':bool(torch.isfinite(y).all()),'grad_finite':bool(torch.isfinite(x.grad).all())})
bench=[]
for counts in dists:
 x=torch.randn(T,H,device="cuda",dtype=torch.float16); probs=torch.ones(T,device="cuda",dtype=torch.float16); tcp=torch.tensor(counts,device="cuda",dtype=torch.int64)
 for _ in range(10): m(x,tcp,probs)
 torch.cuda.synchronize(); st=torch.cuda.Event(True); en=torch.cuda.Event(True); vals=[]
 for _ in range(30): st.record(); m(x,tcp,probs); en.record(); en.synchronize(); vals.append(st.elapsed_time(en)*1000)
 bench.append({'distribution':counts,'mean_us':sum(vals)/len(vals),'min_us':min(vals),'max_us':max(vals),'samples_us':vals})
print(json.dumps({'backend':type(m).__name__,'configs':results,'benchmark':bench,'collectives':0,'dtype':str(next(m.parameters()).dtype),'source':'experts.py SequentialMLP.forward'})); parallel_state.destroy_model_parallel(); dist.destroy_process_group()
'''; enc=base64.b64encode(remote.encode()).decode(); d='/tmp/aka_phase18a'; s.exec_command(f'mkdir -p {d}; echo {enc} | base64 -d > {d}/moe_expert_qualification.py')[1].read(); cmd=f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/moe_expert_qualification.py'; _,o,e=s.exec_command(cmd,timeout=1200); r={'command':cmd,'stdout':o.read().decode(),'stderr':e.read().decode()}; (out/'qualification_raw.json').write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,indent=2)); s.close()
