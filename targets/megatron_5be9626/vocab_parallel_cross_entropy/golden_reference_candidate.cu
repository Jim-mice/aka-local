// PURPOSE: EVALUATOR_VALIDATION_ONLY
// eligible_for_scoring: false
// eligible_for_incumbent: false
// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

__global__ void golden_max(const half* x,float* out,int rows,int lv){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;float m=-3.402823466e+38F;for(int c=0;c<lv;c++)m=fmaxf(m,__half2float(x[(int64_t)r*lv+c]));out[r]=m;}
__global__ void golden_prepare(const half* x,const int64_t* t,const float* gm,float* pred,float* den,float* ev,unsigned char* mask,int64_t* mt,int rows,int lv,int rank){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;int64_t start=(int64_t)rank*lv,end=start+lv,target=t[r];bool own=target>=start&&target<end;mt[r]=own?target-start:0;mask[r]=own?0:1;float p=0.f,s=0.f;for(int c=0;c<lv;c++){float sh=__half2float(x[(int64_t)r*lv+c])-gm[r];float e=expf(sh);ev[(int64_t)r*lv+c]=e;s+=e;if(own&&c==target-start)p=sh;}pred[r]=p;den[r]=s;}
extern "C" cudaError_t local_max_fp16_stream(const half* logits,float* row_max,int rows,int local_vocab,cudaStream_t stream){if(!logits||!row_max||rows<0||local_vocab<=0)return cudaErrorInvalidValue;golden_max<<<(rows+127)/128,128,0,stream>>>(logits,row_max,rows,local_vocab);return cudaGetLastError();}
extern "C" cudaError_t local_prepare_fp16_stream(const half* logits,const int64_t* targets,const float* global_max,float* predicted_local,float* denominator_local,float* exp_values,unsigned char* target_mask,int64_t* target_local,int rows,int local_vocab,int rank,cudaStream_t stream){if(!logits||!targets||!global_max||!predicted_local||!denominator_local||!exp_values||!target_mask||!target_local||rows<0||local_vocab<=0)return cudaErrorInvalidValue;golden_prepare<<<(rows+127)/128,128,0,stream>>>(logits,targets,global_max,predicted_local,denominator_local,exp_values,target_mask,target_local,rows,local_vocab,rank);return cudaGetLastError();}
