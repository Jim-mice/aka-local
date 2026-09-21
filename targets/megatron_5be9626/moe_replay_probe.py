import base64,json,pathlib,paramiko
root=pathlib.Path(__file__).resolve().parents[2]; out=root/'moe_native_sequential_expert_compute'; out.mkdir(parents=True,exist_ok=True); secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
remote='''import json,torch,torch.distributed as dist
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
dist.init_process_group("gloo",init_method="tcp://127.0.0.1:29618",rank=0,world_size=1)
parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,expert_model_parallel_size=1); model_parallel_cuda_manual_seed(123)
E=4; cfg=TransformerConfig(num_layers=1,hidden_size=64,num_attention_heads=4,num_moe_experts=E,moe_ffn_hidden_size=128,use_cpu_initialization=False,activation_func=torch.nn.functional.silu,gated_linear_unit=True,bias_activation_fusion=True,moe_router_load_balancing_type="aux_loss",moe_router_topk=1,moe_router_pre_softmax=True,add_bias_linear=False,params_dtype=torch.float16,fp16=True)
spec=get_submodules(get_gpt_layer_local_submodules(num_experts=E,moe_grouped_gemm=False).mlp); m=MoELayer(cfg,spec).cuda(); x=torch.randn(8,2,64,device="cuda",dtype=torch.float16,requires_grad=True); y,b=m(x); loss=y.float().sum(); loss.backward(); print(json.dumps({'backend':type(m.experts).__name__,'output_shape':list(y.shape),'dtype':str(y.dtype),'finite':bool(torch.isfinite(y).all()),'x_grad_finite':bool(torch.isfinite(x.grad).all()),'experts':E,'tokens':16,'te':False})); parallel_state.destroy_model_parallel(); dist.destroy_process_group()
'''; enc=base64.b64encode(remote.encode()).decode(); d='/tmp/aka_phase18a'; s.exec_command(f'mkdir -p {d}; echo {enc} | base64 -d > {d}/moe_replay_run.py')[1].read(); cmd=f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/moe_replay_run.py'; _,o,e=s.exec_command(cmd,timeout=900); result={'command':cmd,'stdout':o.read().decode(),'stderr':e.read().decode()}; (out/'replay_probe.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); s.close()
