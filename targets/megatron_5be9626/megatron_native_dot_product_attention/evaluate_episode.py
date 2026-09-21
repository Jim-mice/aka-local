import base64, json, paramiko, sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]; epname=sys.argv[1]; ep=root/'campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention'/f'episode_{epname}'; src=ep/'candidate.cu'; secret=__import__('os').environ['AKA_V100_PASSWORD']; ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()); ssh.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20); d='/tmp/aka_phase17b_'+epname.lower(); enc=base64.b64encode(src.read_bytes()).decode(); ssh.exec_command('echo '+enc+' | base64 -d > '+d+'/candidate.cu')[1].read(); _,co,ce=ssh.exec_command('/usr/local/cuda-11.8/bin/nvcc -arch=sm_70 -O3 -shared -Xcompiler -fPIC '+d+'/candidate.cu -o '+d+'/candidate.so',timeout=600); compile_stderr=ce.read().decode();
remote=r'''import ctypes,json,torch,torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
lib=ctypes.CDLL('LIBPATH'); f=lib.dot_product_attention_forward_fp16_stream; f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_float,ctypes.c_void_p]; f.restype=None
def validate(q,k,v,out):
 if q.device.type!='cuda' or k.device!=q.device or v.device!=q.device or out.device!=q.device: raise ValueError('wrong device')
 if q.dtype!=torch.float16 or k.dtype!=torch.float16 or v.dtype!=torch.float16 or out.dtype!=torch.float16: raise TypeError('wrong dtype')
 if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous() or not out.is_contiguous(): raise ValueError('noncontiguous')
 if q.ndim!=4 or k.shape!=q.shape or v.shape!=q.shape or q.shape[2]!=16 or q.shape[3]!=64 or out.shape!=(q.shape[0],q.shape[1],1024): raise ValueError('shape/head mismatch')
def call(q,k,v,out,scale=.125): validate(q,k,v,out); f(q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),q.shape[0],q.shape[1],q.shape[2],q.shape[3],scale,torch.cuda.current_stream().cuda_stream)
dist.init_process_group('gloo',init_method='tcp://127.0.0.1:29598',rank=0,world_size=1); parallel_state._set_global_memory_buffer(); get_cuda_rng_tracker().add('model-parallel-rng',1234); pg=ProcessGroupCollection(); pg.tp=dist.group.WORLD; cfg=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,num_query_groups=16,kv_channels=64,attention_dropout=0.0,attention_softmax_in_fp32=True,masked_softmax_fusion=False,fp16=True,params_dtype=torch.float16); m=DotProductAttention(cfg,1,AttnMaskType.no_mask,'self',pg_collection=pg).cuda(); torch.manual_seed(1717); official=[]
for S,B in [(16,1),(64,2),(128,2)]:
 q=torch.randn(S,B,16,64,device='cuda',dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q); out=torch.empty(S,B,1024,device='cuda',dtype=torch.float16); call(q,k,v,out); torch.cuda.synchronize(); ref=m(q,k,v,None); dd=(out-ref).float().abs(); oracle_q=q.float().permute(1,2,0,3).reshape(B*16,S,64); oracle_k=k.float().permute(1,2,0,3).reshape(B*16,S,64); oracle_v=v.float().permute(1,2,0,3).reshape(B*16,S,64); oracle=torch.bmm(torch.softmax(torch.bmm(oracle_q,oracle_k.transpose(1,2))*.125,dim=-1),oracle_v).reshape(B,16,S,64).permute(2,0,1,3).reshape(S,B,1024).half(); od=(out-oracle).float().abs(); official.append({'shape':[S,B,1024],'max_abs_ref':float(dd.max()),'max_abs_oracle':float(od.max()),'max_rel_oracle':float((od/oracle.float().abs().clamp_min(1e-3)).max()),'finite':bool(torch.isfinite(out).all())})
edges=[]
S,B=16,1; q=torch.zeros(S,B,16,64,device='cuda',dtype=torch.float16); k=torch.ones_like(q); v=torch.full_like(q,2); out=torch.empty(S,B,1024,device='cuda',dtype=torch.float16); call(q,k,v,out); torch.cuda.synchronize(); edges.append({'case':'zero_constant','max_abs':float((out.float()-2).abs().max()),'finite':bool(torch.isfinite(out).all())})
q=torch.full_like(q,-2); k=torch.full_like(k,3); v=torch.randn_like(v); call(q,k,v,out); torch.cuda.synchronize(); edges.append({'case':'negative_near_uniform','finite':bool(torch.isfinite(out).all())})
neg=[]
for name,fn in [('wrong_dtype',lambda:call(q.float(),k,v,out)),('wrong_head_dim',lambda:call(q[:,:,:,:32],k[:,:,:,:32],v[:,:,:,:32],out[:,:,:,:32]))]:
 try: fn(); neg.append({'case':name,'rejected':False})
 except Exception as e: neg.append({'case':name,'rejected':True,'reason':str(e)})
class CF(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v):
  out=torch.empty(q.shape[0],q.shape[1],1024,device='cuda',dtype=torch.float16); call(q,k,v,out); ctx.save_for_backward(q.detach(),k.detach(),v.detach()); return out
 @staticmethod
 def backward(ctx,g):
  q,k,v=ctx.saved_tensors
  with torch.enable_grad():
   qq=q.detach().requires_grad_(); kk=k.detach().requires_grad_(); vv=v.detach().requires_grad_(); yy=m(qq,kk,vv,None); dq,dk,dv=torch.autograd.grad(yy,(qq,kk,vv),g,retain_graph=False)
  return dq,dk,dv
q=torch.randn(16,1,16,64,device='cuda',dtype=torch.float16,requires_grad=True); k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True); g=torch.randn(16,1,1024,device='cuda',dtype=torch.float16); yref=m(q,k,v,None); yref.backward(g); rq,rk,rv=q.grad.detach(),k.grad.detach(),v.grad.detach(); q.grad=None; k.grad=None; v.grad=None; yc=CF.apply(q,k,v); yc.backward(g); compat={'dQ_max_abs':float((rq-q.grad).float().abs().max()),'dK_max_abs':float((rk-k.grad).float().abs().max()),'dV_max_abs':float((rv-v.grad).float().abs().max())}
print(json.dumps({'official':official,'edges':edges,'negative':neg,'collectives':0,'candidate':True,'backward_compatibility':compat})); dist.destroy_process_group()
'''.replace('LIBPATH',d+'/candidate.so')
payload=base64.b64encode(remote.encode()).decode(); ssh.exec_command('echo '+payload+' | base64 -d > '+d+'/evaluate.py')[1].read(); _,o,e=ssh.exec_command('cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python '+d+'/evaluate.py',timeout=900); result={'episode':epname,'compile_stderr':compile_stderr,'stdout':o.read().decode(),'stderr':e.read().decode()[-4000:]}; (ep/'correctness.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); ssh.close()
