#include <cuda_runtime.h>
__global__ void sentinel_probe(float* out) {
    float x = -3.402823466e+38F;
    float y = -INFINITY;
    out[0] = x < y ? x : y;
}
int main() { float* p = nullptr; cudaMallocManaged(&p, sizeof(float)); sentinel_probe<<<1,1>>>(p); cudaDeviceSynchronize(); cudaFree(p); return 0; }
