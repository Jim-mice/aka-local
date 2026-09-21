import base64, hashlib, json, pathlib, paramiko

project=pathlib.Path(__file__).resolve().parents[3]
ep=project/'campaigns/targets/megatron_5be9626/moe_native_sequential_expert_compute/episode_M4'
src=ep/'moe_sequential_expert_forward.cu'
secret=__import__('os').environ['AKA_V100_PASSWORD']
s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
d='/tmp/aka_phase18b_m4_bench'; s.exec_command(f'mkdir -p {d}')[1].read()
enc=base64.b64encode(src.read_bytes()).decode(); s.exec_command(f'echo {enc} | base64 -d > {d}/candidate.cu')[1].read()
_,co,ce=s.exec_command(f'/usr/local/cuda-11.8/bin/nvcc -arch=sm_70 -O3 -shared -Xcompiler -fPIC {d}/candidate.cu -o {d}/candidate.so',timeout=600); cerr=ce.read().decode()
remote=r'''import ctypes,json,torch
lib=ctypes.CDLL('/tmp/aka_phase18b_m4_bench/candidate.so'); f=lib.moe_sequential_expert_forward_fp16_stream
f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
E,H,I,T=4,64,128,16; dists=[[4,4,4,4],[1,3,5,7],[0,0,8,8]]; rows=[]; torch.manual_seed(1818); stream=torch.cuda.current_stream(); fixtures={}
for dist in dists:
 x=torch.randn(T,H,device='cuda',dtype=torch.float16); p=torch.rand(T,device='cuda',dtype=torch.float16); w1=torch.randn(E,2*I,H,device='cuda',dtype=torch.float16)*.02; w2=torch.randn(E,H,I,device='cuda',dtype=torch.float16)*.02; c=torch.tensor(dist,device='cuda',dtype=torch.int64); fixtures[tuple(dist)]=(x,p,w1,w2,c)
def ref(x,p,w1,w2,c,out):
 pos=0
 for e,n in enumerate(c.tolist()):
  if n:
   z=x[pos:pos+n].float(); a=z@w1[e,:I].float().t(); b=z@w1[e,I:].float().t(); out[pos:pos+n].copy_((a/(1+torch.exp(-a))*b)@w2[e].float().t()*p[pos:pos+n].float().unsqueeze(1))
  pos+=n
def cand(x,p,w1,w2,c,out): f(x.data_ptr(),c.data_ptr(),p.data_ptr(),w1.data_ptr(),w2.data_ptr(),out.data_ptr(),T,H,I,E,stream.cuda_stream)
def measure(kind,dist):
 x,p,w1,w2,c=fixtures[tuple(dist)]; out=torch.empty_like(x); st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True); st.record(stream)
 if kind=='reference': ref(x,p,w1,w2,c,out)
 else: cand(x,p,w1,w2,c,out)
 en.record(stream); en.synchronize(); return st.elapsed_time(en)*1000.0
for dist in dists:
 for _ in range(5): measure('reference',dist); measure('candidate',dist)
 for inv in range(3):
  for block in range(3):
   for rep in range(5):
    for kind in ['reference','candidate','candidate','reference']: rows.append({'invocation':inv,'block':block,'sample':rep,'order':kind,'distribution':dist,'latency_us':measure(kind,dist)})
print(json.dumps({'rows':rows}))'''
enc2=base64.b64encode(remote.encode()).decode(); s.exec_command(f'echo {enc2} | base64 -d > {d}/run.py')[1].read(); _,ro,re=s.exec_command(f'<REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/run.py',timeout=1800)
result={'episode':'M4','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'compile_pass':not cerr,'compile_stderr':cerr,'stdout':ro.read().decode(),'stderr':re.read().decode()}; (ep/'paired_benchmark_raw.json').write_text(json.dumps(result)+'\n'); print(json.dumps({'episode':'M4','compile_pass':not cerr,'stdout_bytes':len(result['stdout']),'stderr':result['stderr']},indent=2)); s.close()
