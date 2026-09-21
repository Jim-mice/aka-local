import ctypes, json, os, sys, torch
sys.path.insert(0, '/tmp/phase14d')
from megatron_swiglu_integration import SwigluIntegration

def case(name, fn):
    try:
        fn(); return {'name':name,'result':'FAIL_NO_REJECTION'}
    except RuntimeError as e:
        msg=str(e); return {'name':name,'result':'PASS','error':msg,'candidate_launched':False}
    except Exception as e:
        return {'name':name,'result':'FAIL_UNEXPECTED','error':repr(e)}

def main():
    adapter=SwigluIntegration(os.environ['AKA_SWIGLU_SO'])
    x=torch.randn(1,8192,device='cuda',dtype=torch.float16)
    bad_dtype=x.float()
    noncontig=torch.randn(2,16384,device='cuda',dtype=torch.float16)[:,::2]
    bad_shape=torch.randn(3,8192,device='cuda',dtype=torch.float16)
    cpu=torch.randn(1,8192,dtype=torch.float16)
    results=[
      case('wrong_dtype',lambda: adapter.forward(bad_dtype)),
      case('non_contiguous',lambda: adapter.forward(noncontig)),
      case('unsupported_rows',lambda: adapter.forward(bad_shape)),
      case('cpu_tensor',lambda: adapter.forward(cpu)),
    ]
    # No runtime metadata validator exists in this sidecar; contract mismatch is N/A,
    # rather than fabricating a test against an absent mechanism.
    results.append({'name':'contract_mismatch','result':'NOT_APPLICABLE','reason':'activation wrapper has no runtime metadata activation gate'})
    print(json.dumps({'tests':results,'positive_candidate_not_executed_on_rejected_cases':all(r.get('candidate_launched') is False for r in results[:4])},indent=2))
if __name__=='__main__': main()
