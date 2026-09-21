import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import load_inline

# The installed MSVC emits UTF-8 bytes while this Windows shell reports an
# OEM code page. Make the PyTorch compiler probe decode its own output safely.
cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")


CUDA_SOURCE = r'''
#include <torch/extension.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    const int threads = 256;
    const int blocks = (a.numel() + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
        static_cast<int>(a.numel()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "FP32 vector add");
}
'''


ext = load_inline(
    name="aka_local_vector_add_sm120",
    cpp_sources="",
    cuda_sources=CUDA_SOURCE,
    extra_cflags=["/Zc:preprocessor"],
    extra_cuda_cflags=["-O3", "-arch=sm_120", "-Xcompiler=/Zc:preprocessor"],
    verbose=True,
)

n = 1 << 20
a = torch.randn(n, device="cuda", dtype=torch.float32)
b = torch.randn(n, device="cuda", dtype=torch.float32)
out = torch.empty_like(a)
ref = a + b

for _ in range(20):
    ext.vector_add(a, b, out)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    ext.vector_add(a, b, out)
stop.record()
stop.synchronize()

print("GPU:", torch.cuda.get_device_name())
print("capability:", torch.cuda.get_device_capability())
print("max_abs_error:", (out - ref).abs().max().item())
print("latency_us:", start.elapsed_time(stop) * 1000.0 / 100.0)
