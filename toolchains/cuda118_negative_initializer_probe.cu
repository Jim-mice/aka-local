#include <cuda_runtime.h>
__global__ void probe(float* out) { out[0] = -3.402823466e+38F; }
extern "C" cudaError_t launch_probe(float* out, cudaStream_t stream) { probe<<<1,1,0,stream>>>(out); return cudaGetLastError(); }
