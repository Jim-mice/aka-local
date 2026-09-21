import argparse,ctypes,json,os,statistics,torch,torch.nn as nn,torch.nn.functional as F
from megatron.core.transformer.mlp import MLP,MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
SHAPES=[(16,1,1024),(64,2,1024),(128,2,1024)]
class L(nn.Module):
 def __init__(self,a,b,**k): super().__init__(); self.weight=nn.Parameter(torch.randn(b,a,device='cuda',dtype=torch.float16)*.01)
 def forward(self,x): return F.linear(x,self.weight),None
def model(fused):
 c=TransformerConfig(num_layers=1,hidden_size=1024,num_attention_heads=16,ffn_hidden_size=4096,gated_linear_unit=True,activation_func=F.silu,add_bias_linear=False,bias_activation_fusion=fused,use_te_activation_func=False,tensor_model_parallel_size=1,pipeline_model_parallel_size=1)
 s=MLPSubmodules(linear_fc1=lambda *a,**k:L(a[0],a[1]),linear_fc2=lambda *a,**k:L(a[0],a[1])); return MLP(c,s,input_size=1024,ffn_hidden_size=4096).cuda().half().train()
def ev(fn,n=100):
 for _ in range(10): fn()
 z=[]
 for _ in range(5):
  a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
  for _ in range(n): fn()
  b.record(); torch.cuda.synchronize(); z.append(a.elapsed_time(b)*1000/n)
 return {'mean_us':statistics.mean(z),'std_us':statistics.stdev(z),'cv':statistics.stdev(z)/statistics.mean(z),'samples_us':z}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--forward-so',required=True); ap.add_argument('--backward-so',required=True); a=ap.parse_args()
 fl=ctypes.CDLL(a.forward_so); fl.launch_swiglu_stream.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p]
 bl=ctypes.CDLL(a.backward_so); bl.launch_swiglu_backward_stream.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p]
 torch.manual_seed(14); orig=model(False); integ=model(True); integ.load_state_dict(orig.state_dict()); out=[]
 for S,B,H in SHAPES:
  x=torch.randn(S,B,H,device='cuda',dtype=torch.float16); go=torch.randn(S,B,4096,device='cuda',dtype=torch.float16)
  with torch.no_grad(): z,_=orig.linear_fc1(x); g,u=torch.chunk(z,2,-1); ref=torch.nn.functional.silu(g)*u; act=torch.empty_like(ref); fl.launch_swiglu_stream(z.data_ptr(),0,act.data_ptr(),S*B,8192,0.0,torch.cuda.current_stream().cuda_stream)
  def f1(): orig.linear_fc1(x)
  def af(): torch.nn.functional.silu(g)*u
  def ak(): fl.launch_swiglu_stream(z.data_ptr(),0,act.data_ptr(),S*B,8192,0.0,torch.cuda.current_stream().cuda_stream)
  def f2(): orig.linear_fc2(ref)
  def ab():
   sg=torch.sigmoid(g); torch.cat((go[...,:4096]*(u)*sg*(1+g*(1-sg)),go[...,:4096]*(g*sg)),-1)
  def bk(): bl.launch_swiglu_backward_stream(z.data_ptr(),go[...,:4096].data_ptr(),act.new_empty(z.shape).data_ptr(),S*B,8192,0.0,torch.cuda.current_stream().cuda_stream)
  out.append({'shape':[S,B,H],'fc1_forward':ev(f1),'swiglu_forward_reference':ev(af),'swiglu_forward_candidate':ev(ak),'fc2_forward':ev(f2),'swiglu_backward_reference':ev(ab),'swiglu_backward_candidate':ev(bk)})
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()
