#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// AKA_TARGET_CONTRACT: megatron_torch_rmsnorm_forward:d24ea7a30fb6a91bc1958a955d140992e77e68dad78faa48158f2b907b722731

namespace {

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void rmsnorm_forward_kernel(const __half* __restrict__ x,
                                       const __half* __restrict__ weight,
                                       __half* __restrict__ output,
                                       int64_t rows,
                                       int64_t hidden_size,
                                       float epsilon) {
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int64_t row = static_cast<int64_t>(blockIdx.x); row < rows;
         row += static_cast<int64_t>(gridDim.x)) {
        const __half* row_x = x + row * hidden_size;
        __half* row_output = output + row * hidden_size;

        float sum_squares = 0.0f;
        for (int64_t j = threadIdx.x; j < hidden_size; j += blockDim.x) {
            const float value = __half2float(row_x[j]);
            sum_squares += value * value;
        }
        sum_squares = warp_sum(sum_squares);
        if (lane == 0) {
            warp_sums[warp] = sum_squares;
        }
        __syncthreads();

        float total = (threadIdx.x < 8) ? warp_sums[threadIdx.x] : 0.0f;
        if (warp == 0) {
            total = warp_sum(total);
            if (threadIdx.x == 0) {
                warp_sums[0] = total;
            }
        }
        __syncthreads();

        const float inverse_rms = rsqrtf(warp_sums[0] / static_cast<float>(hidden_size) + epsilon);
        for (int64_t j = threadIdx.x; j < hidden_size; j += blockDim.x) {
            const float value = __half2float(row_x[j]);
            const float scale = __half2float(weight[j]);
            row_output[j] = __float2half_rn(value * inverse_rms * scale);
        }
        __syncthreads();
    }
}

}  // namespace

extern "C" void rmsnorm_forward_fp16_stream(const __half* x,
                                             const __half* weight,
                                             __half* output,
                                             int64_t rows,
                                             int64_t hidden_size,
                                             float epsilon,
                                             cudaStream_t stream) {
    if (rows <= 0 || hidden_size <= 0) {
        return;
    }
    const int64_t max_grid_x = 65535;
    const int grid_x = static_cast<int>(rows < max_grid_x ? rows : max_grid_x);
    rmsnorm_forward_kernel<<<grid_x, 256, 0, stream>>>(
        x, weight, output, rows, hidden_size, epsilon);
}
