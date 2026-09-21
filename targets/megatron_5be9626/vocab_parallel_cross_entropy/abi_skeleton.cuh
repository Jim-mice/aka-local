// CONTRACT SKELETON ONLY - not an implementation and not a candidate.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
extern "C" cudaError_t local_max_fp16_stream(const half* logits, float* row_max, int rows, int local_vocab, cudaStream_t stream);
extern "C" cudaError_t local_prepare_fp16_stream(const half* logits, const int64_t* targets, const float* global_max, float* predicted_local, float* denominator_local, float* exp_values, unsigned char* target_mask, int64_t* target_local, int rows, int local_vocab, int rank, cudaStream_t stream);
