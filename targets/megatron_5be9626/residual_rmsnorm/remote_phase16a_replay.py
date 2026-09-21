import argparse,json,statistics,torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
def err(a,b):
 d=(a.float()-b.float()).abs();return {'max_abs':float(d.max()),'max_rel':float((d/b.float().abs().clamp_min(1e-12)).max())}
def rms_oracle(x,w,e):
 xf=x.float();return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+e)*w.float()).to(x.dtype)
def finite_difference():
 torch.manual_seed(1603);x=torch.randn(2,5,dtype=torch.float64,requires_grad=True);w=torch.randn(5,dtype=torch.float64,requires_grad=True);g=torch.randn_like(x);eps=1e-6;y=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*w;y.backward(g);idx=(0,2);step=1e-6;xp=x.detach().clone();xm=x.detach().clone();xp[idx]+=step;xm[idx]-=step;f=lambda z:(z*torch.rsqrt(z.square().mean(-1,keepdim=True)+eps)*w.detach()*g).sum();fd=float((f(xp)-f(xm))/(2*step));return {'finite_difference_x_index':list(idx),'analytic':float(x.grad[idx]),'numerical':fd,'max_abs':abs(float(x.grad[idx])-fd)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--S',type=int,default=32);p.add_argument('--B',type=int,default=2);p.add_argument('--H',type=int,default=1024);p.add_argument('--mode',choices=['correctness','benchmark','profile'],default='correctness');a=p.parse_args();torch.manual_seed(1601);dev='cuda';cfg=TransformerConfig(num_layers=1,hidden_size=a.H,num_attention_heads=8,normalization='RMSNorm',hidden_dropout=.1,bias_dropout_fusion=True);norm=WrappedTorchNorm(cfg,a.H,eps=cfg.layernorm_epsilon).cuda().half();x=torch.randn(a.S,a.B,a.H,device=dev,dtype=torch.float16,requires_grad=True);res=torch.randn_like(x,requires_grad=True);bias=torch.randn(a.H,device=dev,dtype=torch.float16,requires_grad=True);fn_train=get_bias_dropout_add(True,True);fn_eval=get_bias_dropout_add(False,True)
 if a.mode=='correctness':
  y=norm(x);yo=rms_oracle(x,norm.weight,cfg.layernorm_epsilon);g=torch.randn_like(y);y.backward(g,retain_graph=True);xo=x.detach().float().requires_grad_(True);wo=norm.weight.detach().float().requires_grad_(True);yoo=xo*torch.rsqrt(xo.square().mean(-1,keepdim=True)+cfg.layernorm_epsilon)*wo;yoo.backward(g.float());rms={'forward':err(y,yo),'x_grad':err(x.grad,xo.grad),'weight_grad':err(norm.weight.grad,wo.grad),'x_grad_finite':bool(torch.isfinite(x.grad).all()),'weight_grad_finite':bool(torch.isfinite(norm.weight.grad).all()),'dtype':str(y.dtype),'finite_difference':finite_difference()};torch.manual_seed(1602);bda=fn_train((x,bias),res,.1);torch.manual_seed(1602);mask=(torch.rand_like(x)>=.1).to(x.dtype);bo=res+(x+bias)*mask/.9;bda_eval=fn_eval((x,bias),res,.1);be=res+x+bias;out={'rmsnorm':rms,'bda_train_unqualified_rng_comparison':err(bda,bo),'bda_eval':err(bda_eval,be),'bda_dtypes':{'x':str(x.dtype),'residual':str(res.dtype),'bias':str(bias.dtype)},'training_dropout_rng_note':'torch.rand_like does not reproduce F.dropout Philox mask consumption; this is not a training BDA correctness oracle','source_paths':['WrappedTorchNorm','get_bias_dropout_add'],'te_available':False}
 elif a.mode=='profile':
  for _ in range(10):norm(x)
  torch.cuda.synchronize();torch.cuda.nvtx.range_push('AKA_RMSNORM_BEGIN');torch.cuda.profiler.start()
  for _ in range(20):norm(x)
  torch.cuda.synchronize();torch.cuda.profiler.stop();torch.cuda.nvtx.range_pop();out={'profile':'rmsnorm_only','shape':[a.S,a.B,a.H],'dtype':'float16'}
 else:
  for _ in range(10):norm(x);fn_eval((x,bias),res,.1)
  torch.cuda.synchronize();rows=[]
  for name,call in [('rmsnorm',lambda:norm(x)),('bda_eval',lambda:fn_eval((x,bias),res,.1))]:
   vals=[]
   for i in range(30):
    st=torch.cuda.Event(True);en=torch.cuda.Event(True);st.record();call();en.record();en.synchronize();vals.append(st.elapsed_time(en)*1000)
   rows.append({'boundary':name,'N':30,'mean_us':statistics.mean(vals),'std_us':statistics.stdev(vals),'samples_us':vals})
  out={'benchmark':rows,'shape':[a.S,a.B,a.H],'dtype':'float16'}
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
