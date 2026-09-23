"""H003 attempt 1: native CUDA RMSNorm extension specialized to H=1024."""

import hashlib
import os

import torch
from torch.utils.cpp_extension import load_inline


os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")

CPP = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> rms_fwd(torch::Tensor x, torch::Tensor w, double eps);
std::vector<torch::Tensor> rms_bwd(torch::Tensor g, torch::Tensor x, torch::Tensor w, torch::Tensor r);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rms_fwd);
  m.def("backward", &rms_bwd);
}
"""

CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>

constexpr int H = 1024;
constexpr int T = 256;

__device__ __forceinline__ float block_sum(float v) {
  __shared__ float warp_sums[8];
  for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  v = threadIdx.x < 8 ? warp_sums[lane] : 0.0f;
  if (warp == 0) for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
  if (threadIdx.x == 0) warp_sums[0] = v;
  __syncthreads();
  return warp_sums[0];
}

__global__ void fwd_kernel(const half* x, const half* w, half* y, float* r, int rows, float eps) {
  int row = blockIdx.x;
  float sum = 0.0f;
  #pragma unroll
  for (int j = threadIdx.x; j < H; j += T) { float v = __half2float(x[row*H+j]); sum += v*v; }
  float inv = rsqrtf(block_sum(sum) / float(H) + eps);
  #pragma unroll
  for (int j = threadIdx.x; j < H; j += T)
    y[row*H+j] = __float2half_rn(__half2float(x[row*H+j]) * inv * __half2float(w[j]));
  if (threadIdx.x == 0) r[row] = inv;
}

__global__ void dx_kernel(const half* g, const half* x, const half* w, const float* r, half* dx, int rows) {
  int row = blockIdx.x;
  float dot = 0.0f;
  #pragma unroll
  for (int j = threadIdx.x; j < H; j += T)
    dot += __half2float(g[row*H+j]) * __half2float(w[j]) * __half2float(x[row*H+j]);
  dot = block_sum(dot);
  float inv = r[row];
  float coeff = inv * inv * inv * dot / float(H);
  #pragma unroll
  for (int j = threadIdx.x; j < H; j += T) {
    float gv = __half2float(g[row*H+j]);
    float xv = __half2float(x[row*H+j]);
    float wv = __half2float(w[j]);
    dx[row*H+j] = __float2half_rn(gv*wv*inv - xv*coeff);
  }
}

__global__ void dw_kernel(const half* g, const half* x, const float* r, half* dw, int rows) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= H) return;
  float sum = 0.0f;
  for (int row = 0; row < rows; ++row)
    sum += __half2float(g[row*H+col]) * __half2float(x[row*H+col]) * r[row];
  dw[col] = __float2half_rn(sum);
}

std::vector<torch::Tensor> rms_fwd(torch::Tensor x, torch::Tensor w, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous());
  TORCH_CHECK(x.size(-1) == H && w.numel() == H);
  c10::cuda::CUDAGuard guard(x.device());
  int rows = x.numel() / H;
  auto y = torch::empty_like(x);
  auto r = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  fwd_kernel<<<rows,T,0,stream>>>(reinterpret_cast<half*>(x.data_ptr<at::Half>()), reinterpret_cast<half*>(w.data_ptr<at::Half>()), reinterpret_cast<half*>(y.data_ptr<at::Half>()), r.data_ptr<float>(), rows, float(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y,r};
}

std::vector<torch::Tensor> rms_bwd(torch::Tensor g, torch::Tensor x, torch::Tensor w, torch::Tensor r) {
  c10::cuda::CUDAGuard guard(x.device());
  int rows = x.numel() / H;
  auto dx = torch::empty_like(x);
  auto dw = torch::empty_like(w);
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  dx_kernel<<<rows,T,0,stream>>>(reinterpret_cast<half*>(g.data_ptr<at::Half>()), reinterpret_cast<half*>(x.data_ptr<at::Half>()), reinterpret_cast<half*>(w.data_ptr<at::Half>()), r.data_ptr<float>(), reinterpret_cast<half*>(dx.data_ptr<at::Half>()), rows);
  dw_kernel<<<4,T,0,stream>>>(reinterpret_cast<half*>(g.data_ptr<at::Half>()), reinterpret_cast<half*>(x.data_ptr<at::Half>()), r.data_ptr<float>(), reinterpret_cast<half*>(dw.data_ptr<at::Half>()), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx,dw};
}
"""

_name = "current_agent_rmsnorm_" + hashlib.sha256((CPP + CUDA).encode()).hexdigest()[:12]
_ext = load_inline(name=_name, cpp_sources=CPP, cuda_sources=CUDA, functions=None,
                   extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)


class _NativeRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        y, r = _ext.forward(x, weight, float(eps))
        ctx.save_for_backward(x, weight, r)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, r = ctx.saved_tensors
        dx, dw = _ext.backward(grad_output.contiguous(), x, weight, r)
        return dx, dw, None


class TritonRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size=1024, eps=1.0e-5, num_warps=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)
        self.invocation_count = 0

    def forward(self, x):
        self.invocation_count += 1
        return _NativeRMSNormFunction.apply(x, self.weight, self.eps)
