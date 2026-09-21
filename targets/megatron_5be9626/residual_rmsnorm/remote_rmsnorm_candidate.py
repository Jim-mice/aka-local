import argparse, ctypes, json, statistics, torch
EPS=1e-5
def load(path):
 f=ctypes.CDLL(path).rmsnorm_forward_fp16_stream; f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_float,ctypes.c_void_p]; return f
def call(f,x,w,out):
 f(ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(w.data_ptr()),ctypes.c_void_p(out.data_ptr()),ctypes.c_int64(x.numel()//x.shape[-1]),ctypes.c_int64(x.shape[-1]),ctypes.c_float(EPS),ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
def oracle(x,w): return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+EPS)*w.float()).half()
def check_inputs(x,w,out):
 if x.device.type!='cuda' or w.device!=x.device or out.device!=x.device: raise ValueError('wrong device')
 if x.dtype!=torch.float16 or w.dtype!=torch.float16 or out.dtype!=torch.float16: raise TypeError('wrong dtype')
 if not x.is_contiguous() or not w.is_contiguous() or not out.is_contiguous(): raise ValueError('noncontiguous')
 if x.shape[-1]!=w.numel() or out.shape!=x.shape: raise ValueError('shape mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--lib');p.add_argument('--S',type=int);p.add_argument('--B',type=int);p.add_argument('--H',type=int);p.add_argument('--mode',choices=['official','edge','backward','negative','benchmark'],default='official');p.add_argument('--impl',choices=['reference','candidate'],default='candidate');p.add_argument('--warmup',type=int,default=5);p.add_argument('--blocks',type=int,default=3);p.add_argument('--measurements',type=int,default=10);a=p.parse_args();torch.manual_seed(1616);x=torch.randn(a.S,a.B,a.H,device='cuda',dtype=torch.float16);w=torch.randn(a.H,device='cuda',dtype=torch.float16);out=torch.empty_like(x);f=load(a.lib);norm=torch.nn.RMSNorm(a.H,eps=EPS,device='cuda',dtype=torch.float16);norm.weight.data.copy_(w)
 def candidate(q=x,qq=out): check_inputs(q,w,qq);call(f,q,w,qq);return qq
 if a.mode=='official':
  y=candidate();torch.cuda.synchronize();r=oracle(x,w);d=(y.float()-r.float()).abs();print(json.dumps({'max_abs':float(d.max()),'max_rel':float((d/r.float().abs().clamp_min(1e-12)).max()),'finite':bool(torch.isfinite(y).all())}));return
 if a.mode=='edge':
  cases={'zero':torch.zeros_like(x),'constant':torch.full_like(x,.5),'positive':torch.full_like(x,2),'negative':torch.full_like(x,-2),'small':torch.full_like(x,1e-4),'large':torch.full_like(x,100)};ans=[]
  for n,q in cases.items():
   z=torch.empty_like(q);candidate(q,z);torch.cuda.synchronize();r=oracle(q,w);d=(z.float()-r.float()).abs();ans.append({'case':n,'max_abs':float(d.max()),'max_rel':float((d/r.float().abs().clamp_min(1e-12)).max()),'finite':bool(torch.isfinite(z).all())})
  print(json.dumps({'edge':ans}));return
 if a.mode=='negative':
  tests={'wrong_x_dtype':(x.float(),w,out),'wrong_w_dtype':(x,w.float(),out),'noncontig_x':(x.transpose(0,1),w,out),'noncontig_w':(x,w.as_strided((a.H,),(0,)),out),'wrong_output':(x,w,out[:,:,:a.H-1])};ans=[]
  for n,v in tests.items():
   try: check_inputs(*v);ans.append({'case':n,'rejected':False,'kernel_launched':False})
   except Exception as e: ans.append({'case':n,'rejected':True,'reason':str(e),'kernel_launched':False})
  print(json.dumps({'negative':ans}));return
 if a.mode=='backward':
  xx=x.detach().requires_grad_(True);g=torch.randn_like(x);ref=torch.nn.RMSNorm(a.H,eps=EPS,device='cuda',dtype=torch.float16);ref.weight.data.copy_(w);yy=ref(xx);yy.backward(g);rx=xx.grad.detach();rw=ref.weight.grad.detach();sx=x.detach().requires_grad_(True);side=torch.nn.RMSNorm(a.H,eps=EPS,device='cuda',dtype=torch.float16);side.weight.data.copy_(w);side(sx).backward(g);dx=(rx.float()-sx.grad.float()).abs();dw=(rw.float()-side.weight.grad.float()).abs();print(json.dumps({'x_grad_max_abs':float(dx.max()),'x_grad_max_rel':float((dx/rx.float().abs().clamp_min(1e-12)).max()),'weight_grad_max_abs':float(dw.max()),'weight_grad_max_rel':float((dw/rw.float().abs().clamp_min(1e-12)).max()),'architecture':'candidate forward plus trusted reference backward sidecar'}));return
 for _ in range(a.warmup): norm(x) if a.impl=='reference' else candidate()
 torch.cuda.synchronize();rows=[]
 for b in range(a.blocks):
  for i in range(a.measurements):
   s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record();norm(x) if a.impl=='reference' else candidate();e.record();e.synchronize();rows.append(float(s.elapsed_time(e)*1000))
 print(json.dumps({'impl':a.impl,'rows':rows,'mean_us':statistics.mean(rows)}))
if __name__=='__main__':main()
