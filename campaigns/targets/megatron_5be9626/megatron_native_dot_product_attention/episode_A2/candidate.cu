#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: megatron_native_dot_product_attention_forward:90839e70b4edbfdc9cdfd903fe9c51230311a8fc7c13fb1b2a0673acd261916c

namespace {
__global__ void native_attention_kernel(const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v, __half* __restrict__ out, int S, int B, int H, int D, float scale) {
    extern __shared__ float score[];
    const int row = blockIdx.x, qpos = row % S, bh = row / S, b = bh / H, h = bh % H, tid = threadIdx.x;
    if (b >= B || h >= H || qpos >= S) return;
    const int qbase = ((qpos * B + b) * H + h) * D;
    for (int spos = tid; spos < S; spos += blockDim.x) {
        const int kbase = ((spos * B + b) * H + h) * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += __half2float(q[qbase + d]) * __half2float(k[kbase + d]);
        score[spos] = dot * scale;
    }
    __syncthreads();
    if (tid == 0) {
        const float neg_max = -3.402823466e+38F;
        float mx = neg_max;
        for (int spos = 0; spos < S; ++spos) if (score[spos] > mx) mx = score[spos];
        float sum = 0.0f;
        for (int spos = 0; spos < S; ++spos) { score[spos] = expf(score[spos] - mx); sum += score[spos]; }
        const float inv_sum = 1.0f / sum;
        for (int spos = 0; spos < S; ++spos) score[spos] *= inv_sum;
    }
    __syncthreads();
    const int obase = ((qpos * B + b) * H * D) + h * D;
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int spos = 0; spos < S; ++spos) {
            const int vbase = ((spos * B + b) * H + h) * D;
            acc += score[spos] * __half2float(v[vbase + d]);
        }
        out[obase + d] = __float2half_rn(acc);
    }
}
}

extern "C" void dot_product_attention_forward_fp16_stream(const __half* q, const __half* k, const __half* v, __half* output, int64_t seq_len, int64_t batch, int64_t num_heads, int64_t head_dim, float scale, cudaStream_t stream) {
    const int S = static_cast<int>(seq_len), B = static_cast<int>(batch), H = static_cast<int>(num_heads), D = static_cast<int>(head_dim);
    if (S <= 0 || B <= 0 || H <= 0 || D <= 0) return;
    native_attention_kernel<<<dim3(static_cast<unsigned int>(S * B * H)), dim3(128), static_cast<size_t>(S) * sizeof(float), stream>>>(q, k, v, output, S, B, H, D, scale);
}
