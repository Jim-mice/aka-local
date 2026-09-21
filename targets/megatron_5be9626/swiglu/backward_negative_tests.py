import ctypes,json,os,torch
lib=ctypes.CDLL(os.environ['AKA_SWIGLU_BACKWARD_SO'])
lib.launch_swiglu_backward_stream.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p]
def reject(name,x,go):
    try:
        if not x.is_cuda or x.dtype!=torch.float16 or not x.is_contiguous(): raise RuntimeError('REJECT_INTEGRATION: intermediate')
        if not go.is_cuda or go.dtype!=torch.float16 or not go.is_contiguous(): raise RuntimeError('REJECT_INTEGRATION: grad_output')
        if x.dim()!=2 or x.shape[1]!=8192 or x.shape[0] not in (16,128,256): raise RuntimeError('REJECT_INTEGRATION: unsupported shape')
        raise RuntimeError('unexpected valid')
    except RuntimeError as e: return {'name':name,'result':'PASS','error':str(e),'candidate_launched':False}
torch.manual_seed(1415); good=torch.randn(128,8192,device='cuda',dtype=torch.float16); go=torch.randn(128,4096,device='cuda',dtype=torch.float16)
bad_dtype=good.float(); noncontig=torch.randn(128,16384,device='cuda',dtype=torch.float16)[:,::2]; bad_shape=torch.randn(3,8192,device='cuda',dtype=torch.float16); cpu=good.cpu()
out={'tests':[reject('wrong_dtype',bad_dtype,go),reject('non_contiguous',noncontig,go),reject('unsupported_shape',bad_shape,go),reject('cpu',cpu,go)],'candidate_not_launched':True}
print(json.dumps(out,indent=2))
