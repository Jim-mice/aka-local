#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: megatron_torch_rmsnorm_forward:d24ea7a30fb6a91bc1958a955d140992e77e68dad78faa48158f2b907b722731
extern "C" void rmsnorm_forward_fp16_stream(const __half* x, const __half* weight, __half* output, int64_t rows, int64_t hidden_size, float epsilon, cudaStream_t stream);
