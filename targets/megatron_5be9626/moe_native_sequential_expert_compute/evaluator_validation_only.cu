#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
// EVALUATOR_VALIDATION_ONLY NOT_AGENT_CANDIDATE NOT_BENCHMARK_ELIGIBLE NOT_INCUMBENT_ELIGIBLE
// AKA_TARGET_CONTRACT: a1bb8ee0af6c250663e0935ae3deb07104975fc5904c56bb5b2f0d4e8032d1fe
extern "C" void moe_sequential_expert_forward_fp16_stream(const half* tokens, const int64_t* tokens_per_expert, const half* probs, const half* fc1_weights, const half* fc2_weights, half* output, int64_t token_count, int64_t hidden, int64_t intermediate, int64_t experts, cudaStream_t stream) {
    (void)tokens; (void)tokens_per_expert; (void)probs; (void)fc1_weights; (void)fc2_weights; (void)output;
    (void)token_count; (void)hidden; (void)intermediate; (void)experts; (void)stream;
}
