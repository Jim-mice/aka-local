#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: megatron_native_dot_product_attention_forward:90839e70b4edbfdc9cdfd903fe9c51230311a8fc7c13fb1b2a0673acd261916c

namespace {
__global__ void attention_forward_kernel(const __half* q, const __half* k,
    const __half* v, __half* output, int S, int B, int H, int D, float scale) {
    extern __shared__ unsigned char raw[];
    __half* qs = reinterpret_cast<__half*>(raw);
    float* p = reinterpret_cast<float*>(raw + 64 * sizeof(__half));
    __shared__ float reduce[128];
    __shared__ float mx;
    __shared__ float denom;
    int tid = threadIdx.x, id = blockIdx.x, h = id % H;
    int b = (id / H) % B, s = id / (H * B);
    int base = ((s * B + b) * H + h) * D;
    for (int d = tid; d < D; d += blockDim.x) qs[d] = q[base + d];
    __syncthreads();
    float local_max = -CUDART_INF_F;
    for (int ks = tid; ks < S; ks += blockDim.x) {
        int kb = ((ks * B + b) * H + h) * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += __half2float(qs[d]) * __half2float(k[kb + d]);
        p[ks] = dot * scale;
        local_max = fmaxf(local_max, p[ks]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int n = 64; n; n >>= 1) { if (tid < n) reduce[tid] = fmaxf(reduce[tid], reduce[tid+n]); __syncthreads(); }
    if (tid == 0) mx = reduce[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (int ks = tid; ks < S; ks += blockDim.x) { p[ks] = expf(p[ks] - mx); local_sum += p[ks]; }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int n = 64; n; n >>= 1) { if (tid < n) reduce[tid] += reduce[tid+n]; __syncthreads(); }
    if (tid == 0) denom = reduce[0];
    __syncthreads();
    for (int d = tid; d < D; d += blockDim.x) {
        float out = 0.0f;
        for (int ks = 0; ks < S; ++ks) {
            int vb = ((ks * B + b) * H + h) * D;
            out += (p[ks] / denom) * __half2float(v[vb + d]);
        }
        output[base + d] = __float2half(out);
    }
}
}

extern "C" void dot_product_attention_forward_fp16_stream(
    const __half* q, const __half* k, const __half* v, __half* output,
    int64_t seq_len, int64_t batch, int64_t num_heads, int64_t head_dim,
    float scale, cudaStream_t stream) {
    size_t smem = 64 * sizeof(__half) + static_cast<size_t>(seq_len) * sizeof(float);
    int blocks = static_cast<int>(seq_len * batch * num_heads);
    attention_forward_kernel<<<blocks, 128, smem, stream>>>(q, k, v, output,
        static_cast<int>(seq_len), static_cast<int>(batch), static_cast<int>(num_heads),
        static_cast<int>(head_dim), scale);
}
