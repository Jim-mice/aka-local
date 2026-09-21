#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

// AKA_TARGET_CONTRACT: megatron_native_dot_product_attention_forward:90839e70b4edbfdc9cdfd903fe9c51230311a8fc7c13fb1b2a0673acd261916c

namespace {
__device__ __forceinline__ float bsum(float x, float* r) { int t=threadIdx.x; r[t]=x; __syncthreads(); for(int n=blockDim.x>>1;n;n>>=1){if(t<n)r[t]+=r[t+n];__syncthreads();} return r[0]; }
__device__ __forceinline__ float bmax(float x, float* r) { int t=threadIdx.x; r[t]=x; __syncthreads(); for(int n=blockDim.x>>1;n;n>>=1){if(t<n&&r[t+n]>r[t])r[t]=r[t+n];__syncthreads();} return r[0]; }
__global__ void row(const __half* q,const __half* k,const __half* v,__half* o,int64_t s,int64_t b,int64_t h,int64_t d,float scale) {
  int i=blockIdx.x, bb=blockIdx.y, hh=blockIdx.z, t=threadIdx.x, S=(int)s,D=(int)d,B=(int)b,H=(int)h;
  extern __shared__ float sm[]; float* score=sm; float* red=sm+S;
  int64_t qb=(((int64_t)i*B+bb)*H+hh)*D;
  for(int j=0;j<S;++j){ float x=0; int64_t kb=(((int64_t)j*B+bb)*H+hh)*D; for(int z=t;z<D;z+=blockDim.x)x+=__half2float(q[qb+z])*__half2float(k[kb+z]); float sum=bsum(x,red); if(!t)score[j]=sum*scale; __syncthreads(); }
  float lm=-FLT_MAX; for(int j=t;j<S;j+=blockDim.x)lm=fmaxf(lm,score[j]); float mx=bmax(lm,red);
  float ls=0; for(int j=t;j<S;j+=blockDim.x){float e=expf(score[j]-mx);score[j]=e;ls+=e;} float den=bsum(ls,red);
  if(t<D){float acc=0; for(int j=0;j<S;++j){int64_t vb=(((int64_t)j*B+bb)*H+hh)*D;acc+=(score[j]/den)*__half2float(v[vb+t]);} int64_t ob=((int64_t)i*B+bb)*H*D+(int64_t)hh*D;o[ob+t]=__float2half_rn(acc);}
}
}
extern "C" void dot_product_attention_forward_fp16_stream(const __half* q,const __half* k,const __half* v,__half* output,int64_t seq_len,int64_t batch,int64_t num_heads,int64_t head_dim,float scale,cudaStream_t stream){
  const int threads=128; size_t shared=(size_t)seq_len*sizeof(float)+(size_t)threads*sizeof(float); dim3 grid((unsigned)seq_len,(unsigned)batch,(unsigned)num_heads); row<<<grid,threads,shared,stream>>>(q,k,v,output,seq_len,batch,num_heads,head_dim,scale);
}
