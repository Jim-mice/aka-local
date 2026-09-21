#include <cuda_fp16.h>
#include <cuda_runtime.h>

// AKA_TARGET_INTEGRATION: swiglu_backward_v1 current_stream adapter_from=95f66fb9cbf5f4ef
__device__ __forceinline__ half sigf(half x) { float v=__half2float(x); return __float2half(1.0f/(1.0f+expf(-v))); }
__global__ void k(const half* x,const half* go,half* gi,int rows,int width,half off) {
  int hw=width>>1, i=blockIdx.x*blockDim.x+threadIdx.x, n=rows*hw; if(i>=n)return;
  int c=i%hw, base=(i/hw)*width+c; half g=x[base],u=x[base+hw],d=go[i],s=sigf(g);
  half silu=__hmul(g,s), dg=__hmul(__hmul(s,__hadd(__float2half(1.0f),__hmul(g,__hsub(__float2half(1.0f),s)))),__hadd(u,off));
  gi[base]=__hmul(d,dg); gi[base+hw]=__hmul(d,silu);
}
extern "C" void launch_swiglu_backward_stream(const half* x,const half* go,half* gi,int rows,int width,float offset,cudaStream_t stream) {
  int n=rows*(width>>1); k<<<(n+255)/256,256,0,stream>>>(x,go,gi,rows,width,__float2half(offset));
}
