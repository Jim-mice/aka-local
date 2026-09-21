#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b

namespace {

__global__ void local_max_kernel(const half* logits, float* output, int rows, int vocab) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const half* x = logits + static_cast<size_t>(row) * vocab;
    float m = -INFINITY;
    for (int i = 0; i < vocab; ++i) m = fmaxf(m, __half2float(x[i]));
    output[row] = m;
}

__global__ void local_prepare_kernel(const half* logits, const int64_t* targets,
                                     const float* global_max, float* predicted_partial,
                                     float* denominator_partial, float* target_logit_partial,
                                     unsigned char* target_is_local, int64_t* local_target,
                                     int rows, int vocab, int64_t vocab_offset) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const half* x = logits + static_cast<size_t>(row) * vocab;
    float m = global_max[row];
    int64_t target = targets[row];
    int64_t local = target - vocab_offset;
    bool inside = local >= 0 && local < vocab;
    float target_value = 0.0f;
    float denom = 0.0f;
    for (int i = 0; i < vocab; ++i) {
        float e = expf(__half2float(x[i]) - m);
        denom += e;
        if (inside && i == local) target_value = e;
    }
    predicted_partial[row] = target_value;
    denominator_partial[row] = denom;
    target_logit_partial[row] = target_value;
    target_is_local[row] = inside ? 1 : 0;
    local_target[row] = inside ? local : -1;
}

}

extern "C" void local_max_fp16(const half* logits, float* output, int rows, int vocab) {
    constexpr int threads = 128;
    local_max_kernel<<<(rows + threads - 1) / threads, threads>>>(logits, output, rows, vocab);
}

extern "C" void local_prepare_fp16(const half* logits, const int64_t* targets,
                                    const float* global_max, float* predicted_partial,
                                    float* denominator_partial, float* target_logit_partial,
                                    unsigned char* target_is_local, int64_t* local_target,
                                    int rows, int vocab, int64_t vocab_offset) {
    constexpr int threads = 128;
    local_prepare_kernel<<<(rows + threads - 1) / threads, threads>>>(
        logits, targets, global_max, predicted_partial, denominator_partial,
        target_logit_partial, target_is_local, local_target, rows, vocab, vocab_offset);
}
