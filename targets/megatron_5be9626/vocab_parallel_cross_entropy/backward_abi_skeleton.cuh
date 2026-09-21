// Evaluator ABI skeleton only. It is not a candidate and must not be scored.
// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy_backward:6121f49401f3ef4601549c8732a62870ff17a7a7d7047e9c55d55b79c9ff6e92
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" void ce_backward_local_fp32_stream(
    const float* softmax,
    const bool* target_mask,
    const int64_t* masked_target_1d,
    const float* grad_output,
    float* grad_input,
    int64_t rows,
    int64_t local_vocab,
    cudaStream_t stream);
