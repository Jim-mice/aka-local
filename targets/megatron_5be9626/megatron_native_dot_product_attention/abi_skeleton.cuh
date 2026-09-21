#pragma once
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: megatron_native_dot_product_attention_forward:90839e70b4edbfdc9cdfd903fe9c51230311a8fc7c13fb1b2a0673acd261916c
extern "C" void dot_product_attention_forward_fp16_stream(
    const __half* q,
    const __half* k,
    const __half* v,
    __half* output,
    int64_t seq_len,
    int64_t batch,
    int64_t num_heads,
    int64_t head_dim,
    float scale,
    cudaStream_t stream);
