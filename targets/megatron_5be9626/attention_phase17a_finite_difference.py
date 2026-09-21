import torch
torch.manual_seed(17)
q=torch.randn(2,1,2,2,dtype=torch.float64,requires_grad=True); k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True)
def f(q,k,v):
    q=q.permute(1,2,0,3).reshape(2,2,2); k=k.permute(1,2,0,3).reshape(2,2,2); v=v.permute(1,2,0,3).reshape(2,2,2)
    return torch.bmm(torch.softmax(torch.bmm(q,k.transpose(1,2))/2**0.5,dim=-1),v).sum()
y=f(q,k,v); y.backward(); h=1e-6; z=q.detach().clone(); z[0,0,0,0]+=h; zp=f(z,k.detach(),v.detach()); z[0,0,0,0]-=2*h; zm=f(z,k.detach(),v.detach()); print(float(abs((zp-zm)/(2*h)-q.grad[0,0,0,0])))
