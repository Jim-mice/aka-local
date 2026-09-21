#include <cuda_fp16.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b

namespace {

__global__ void local_max_kernel(const half* __restrict__ logits,
                                 float* __restrict__ row_max,
                                 int rows, int vocab) {
    int row = blockIdx.x;
    if (row >= rows) return;
    float best = -CUDART_INF_F;
    const half* x = logits + (size_t)row * vocab;
    for (int col = threadIdx.x; col < vocab; col += blockDim.x) {
        best = fmaxf(best, __half2float(x[col]));
    }
    extern __shared__ float smem[];
    smem[threadIdx.x] = best;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) row_max[row] = smem[0];
}

__global__ void local_prepare_kernel(const half* __restrict__ logits,
                                     const int64_t* __restrict__ targets,
                                     const float* __restrict__ global_max,
                                     float* __restrict__ local_predicted,
                                     float* __restrict__ local_denominator,
                                     float* __restrict__ local_loss,
                                     unsigned char* __restrict__ target_is_local,
                                     int64_t* __restrict__ local_target,
                                     int rows, int local_vocab, int vocab_offset) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const half* x = logits + (size_t)row * local_vocab;
    float m = global_max[row];
    float denom = 0.0f;
    for (int col = 0; col < local_vocab; ++col) denom += expf(__half2float(x[col]) - m);
    int64_t target = targets[row];
    bool owned = target >= vocab_offset && target < (int64_t)vocab_offset + local_vocab;
    float target_logit = 0.0f;
    if (owned) target_logit = __half2float(x[target - vocab_offset]);
    local_predicted[row] = owned ? target_logit : 0.0f;
    local_denominator[row] = denom;
    local_loss[row] = owned ? (m - target_logit) : 0.0f;
    target_is_local[row] = owned ? 1 : 0;
    local_target[row] = owned ? target - vocab_offset : -1;
}

} // namespace

extern "C" void local_max_fp16(const half* logits, float* row_max, int rows, int local_vocab) {
    constexpr int threads = 256;
    local_max_kernel<<<rows, threads, threads * sizeof(float)>>>(logits, row_max, rows, local_vocab);
}

extern "C" void local_prepare_fp16(const half* logits, const int64_t* targets,
                                    const float* global_max, float* local_predicted,
                                    float* local_denominator, float* local_loss,
                                    unsigned char* target_is_local, int64_t* local_target,
                                    int rows, int local_vocab, int vocab_offset) {
    constexpr int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    local_prepare_kernel<<<blocks, threads>>>(logits, targets, global_max, local_predicted,
                                              local_denominator, local_loss, target_is_local,
                                              local_target, rows, local_vocab, vocab_offset);
}
