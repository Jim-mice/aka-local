import json, torch
def f(x,go):
    g,u=torch.chunk(x,2,-1); return (torch.nn.functional.silu(g)*u*go).sum()
torch.manual_seed(1414); x=torch.randn(2,8,dtype=torch.float32,requires_grad=True); go=torch.randn(2,4,dtype=torch.float32)
y=f(x,go); (a,)=torch.autograd.grad(y,x)
checks=[]
for idx in [(0,0),(0,5),(1,7)]:
    eps=1e-3; xp=x.detach().clone(); xm=x.detach().clone(); xp[idx]+=eps; xm[idx]-=eps
    num=(f(xp,go)-f(xm,go))/(2*eps); checks.append({'index':list(idx),'analytical':float(a[idx]),'finite_difference':float(num),'abs_error':float(abs(a[idx]-num))})
out={'dtype':'float32','checks':checks,'max_abs_error':max(c['abs_error'] for c in checks),'pass':max(c['abs_error'] for c in checks)<2e-3}
print(json.dumps(out,indent=2))
