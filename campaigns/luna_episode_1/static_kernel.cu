#include <cuda_fp16.h>

__device__ __forceinline__ float silu_static(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swiglu_static(const half* gate, const half* up, half* out, long n) {
    long pair = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long pairs = n / 2;
    if (pair < pairs) {
        float2 g = __half22float2(reinterpret_cast<const half2*>(gate)[pair]);
        float2 u = __half22float2(reinterpret_cast<const half2*>(up)[pair]);
        reinterpret_cast<half2*>(out)[pair] = __floats2half2_rn(silu_static(g.x) * u.x, silu_static(g.y) * u.y);
    }
}
