import base64, hashlib, json, pathlib, paramiko

project = pathlib.Path(__file__).resolve().parents[3]
ep = project / 'campaigns/targets/megatron_5be9626/moe_native_sequential_expert_compute/episode_M4'
src = ep / 'moe_sequential_expert_forward.cu'
secret = __import__("os").environ["AKA_V100_PASSWORD"]
s = paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect('<REMOTE_HOST>', username='<REMOTE_USER>', password=secret, timeout=20)
d = '/tmp/aka_phase18b_m4'
s.exec_command(f'mkdir -p {d}')[1].read()
enc = base64.b64encode(src.read_bytes()).decode()
s.exec_command(f'echo {enc} | base64 -d > {d}/candidate.cu')[1].read()
_, co, ce = s.exec_command(f'/usr/local/cuda-11.8/bin/nvcc -arch=sm_70 -O3 -shared -Xcompiler -fPIC {d}/candidate.cu -o {d}/candidate.so', timeout=600)
compile_err = ce.read().decode()
remote = r'''import ctypes,json,torch
lib=ctypes.CDLL('/tmp/aka_phase18b_m4/candidate.so')
f=lib.moe_sequential_expert_forward_fp16_stream
f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_void_p]
E,H,I,T=4,64,128,16; results=[]
torch.manual_seed(1804)
for counts in ([4,4,4,4],[1,3,5,7],[0,0,8,8]):
 x=torch.randn(T,H,device='cuda',dtype=torch.float16); probs=torch.rand(T,device='cuda',dtype=torch.float16); w1=torch.randn(E,2*I,H,device='cuda',dtype=torch.float16)*.02; w2=torch.randn(E,H,I,device='cuda',dtype=torch.float16)*.02; out=torch.empty_like(x); c=torch.tensor(counts,device='cuda',dtype=torch.int64)
 f(x.data_ptr(),c.data_ptr(),probs.data_ptr(),w1.data_ptr(),w2.data_ptr(),out.data_ptr(),T,H,I,E,torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize(); oracle=torch.empty_like(x.float()); pos=0
 for e,n in enumerate(counts):
  if n:
   z=x[pos:pos+n].float(); a=z@w1[e,:I].float().t(); b=z@w1[e,I:].float().t(); oracle[pos:pos+n]=(a/(1+torch.exp(-a))*b)@w2[e].float().t()*probs[pos:pos+n].float().unsqueeze(1)
  pos+=n
 err=(out.float()-oracle).abs(); results.append({'distribution':counts,'max_abs':float(err.max()),'max_rel':float((err/oracle.abs().clamp_min(1e-3)).max()),'finite':bool(torch.isfinite(out).all())})
print(json.dumps({'results':results}))'''
enc2 = base64.b64encode(remote.encode()).decode()
s.exec_command(f'echo {enc2} | base64 -d > {d}/evaluate.py')[1].read()
_, ro, re = s.exec_command(f'<REMOTE_HOME>/venvs/lerobot-act/bin/python {d}/evaluate.py', timeout=900)
result={'episode':'M4','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'compile_pass':not compile_err,'compile_stderr':compile_err,'stdout':ro.read().decode(),'stderr':re.read().decode()}
(ep/'build_correctness.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2)); s.close()
