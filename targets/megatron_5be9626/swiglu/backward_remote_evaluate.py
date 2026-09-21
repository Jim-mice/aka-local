import argparse,ctypes,json,math,subprocess,statistics,torch
SHAPES=[(16,1,1024),(64,2,1024),(128,2,1024)]; NVCC='/usr/local/cuda-11.8/bin/nvcc'; CONTRACT='a7437b0c42289669'
def analytical(x,go):
    gate,up=torch.chunk(x,2,-1); sig=torch.sigmoid(gate); silu=gate*sig; ds=sig*(1+gate*(1-sig)); return torch.cat((go*(up)*ds,go*silu),-1)
def stats(fn):
    for _ in range(10): fn()
    out=[]
    for _ in range(5):
        a,b=torch.cuda.Event(True),torch.cuda.Event(True); a.record()
        for _ in range(100): fn()
        b.record(); torch.cuda.synchronize(); out.append(a.elapsed_time(b)*1000/100)
    m=statistics.mean(out); s=statistics.stdev(out); return {'mean_us':m,'std_us':s,'cv':s/m,'samples_us':out}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidate',required=True); ap.add_argument('--profile',action='store_true'); a=ap.parse_args(); so='/tmp/aka_swiglu_backward.so'
    cp=subprocess.run([NVCC,'-Xcompiler','-fPIC','-shared','-O2','-gencode','arch=compute_70,code=sm_70','-o',so,a.candidate],capture_output=True,text=True)
    result={'compile_pass':cp.returncode==0,'compile_error':cp.stderr[-3000:],'contract_hash':CONTRACT,'shapes':[]}
    if cp.returncode: print(json.dumps(result,indent=2)); return
    lib=ctypes.CDLL(so); lib.launch_swiglu_backward.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p]
    torch.manual_seed(1400)
    for S,B,H in SHAPES:
        x=torch.randn(S*B,8192,device='cuda',dtype=torch.float16); go=torch.randn(S*B,4096,device='cuda',dtype=torch.float16); ref=analytical(x,go); out=torch.empty_like(x)
        def cand(): lib.launch_swiglu_backward(x.data_ptr(),go.data_ptr(),out.data_ptr(),S*B,8192,0.0,torch.cuda.current_stream().cuda_stream)
        cand(); torch.cuda.synchronize(); e=(out-ref).abs(); rel=(e/(ref.abs()+1e-3)).max()
        row={'shape':[S,B,H],'max_abs_error':float(e.max()),'max_rel_error':float(rel),'correctness':bool(torch.allclose(out,ref,atol=2e-3,rtol=2e-3)),'baseline':stats(lambda: analytical(x,go)),'candidate':stats(cand)}
        row['baseline_latency_us']=row['baseline']['mean_us']; row['candidate_latency_us']=row['candidate']['mean_us']; row['speedup']=row['baseline_latency_us']/row['candidate_latency_us']; result['shapes'].append(row)
    result['correctness_pass']=all(r['correctness'] for r in result['shapes']); result['geometric_mean_speedup']=math.prod(r['speedup'] for r in result['shapes'])**(1/len(result['shapes'])); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
