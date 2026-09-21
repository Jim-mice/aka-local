"""Correctness-only stage summary helpers; never used by campaign scoring."""
import hashlib, json
def summary(name, tensor):
    x=tensor.detach().float().reshape(-1)
    return {"stage":name,"shape":list(tensor.shape),"dtype":str(tensor.dtype),"min":float(x.min()),"max":float(x.max()),"mean":float(x.mean()),"selected":x[:8].tolist(),"checksum":hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]}
def first_divergence(expected, actual, atol, rtol):
    import torch
    for name in expected:
        e,a=expected[name],actual[name]; err=(a.float()-e.float()).abs(); rel=err/(e.float().abs()+1e-6)
        if not torch.allclose(a,e,atol=atol,rtol=rtol):
            return {"FIRST_DIVERGENCE_STAGE":name,"max_abs":float(err.max()),"max_rel":float(rel.max())}
    return {"FIRST_DIVERGENCE_STAGE":None}
