# Phase 14-F — SwiGLU training overhead and FC1 boundary analysis

## Conclusion

The `0.9071x` integrated training result is not a failure of the CUDA
activation kernels. The raw saved-Python activation measurements show real
speedups, but the complete TP=1 path is dominated by GEMMs and by integration
graph/allocation overhead. The current sidecar is therefore not promoted as a
net training optimization. A future FC1-plus-SwiGLU epilogue is technically
interesting, but remains research-only and has materially greater backward
complexity.

## 1. Timing boundaries and equivalence

The Phase 14-E benchmark times, for both paths, one complete `MLP.forward` plus
one `backward(grad_output)`, with model construction, random generation,
warmup, and optimizer work outside timing. Both paths use identical TP=1 local
linear modules, weights, FP16 inputs, upstream gradients, and gradient reset.

The original path uses Megatron's ordinary SwiGLU branch and autograd. The
integrated path uses the same `MLP.forward`, real FC1/FC2, Episode 2 forward,
and Episode 2 backward through a custom autograd Function. No FC1/FC2 operation
was replaced. CUDA-event placement surrounds the same Python iteration body.

The integration path allocates an activation output and a gradient output each
call, and saves the FC1 intermediate for backward. The original path also
materializes activation intermediates required by PyTorch autograd, but its
saved representation is implementation-dependent. No explicit global sync,
`.contiguous()` copy, repeated compilation, or repeated library load is in the
timed body after the Phase 14-E library-load cache fix.

## 2. Component measurements

Fresh V100 component measurements used the same official shapes and CUDA-event
methodology. Values below are representative at `[128,2,1024]`; isolated
operation timings include their own operation-level launch/allocation behavior
and must not be arithmetically substituted for the complete MLP benchmark.

| Component | Reference us | Candidate us |
|---|---:|---:|
| FC1 forward GEMM | 344.3 | unchanged |
| SwiGLU forward | 89.5 | 59.7 |
| FC2 forward GEMM | 212.7 | unchanged |
| SwiGLU backward | 294.4 | 89.1 |

At smaller shapes the same experiment showed FC1/FC2 GEMM costs of roughly
104/89 us at `[16,1,1024]` and 214/167 us at `[64,2,1024]`. The backward
candidate replaces multiple PyTorch elementwise kernels with one CUDA kernel.

The complete training rerun was:

- original: `2133.61 us`
- integrated: `2352.13 us`
- integrated/original: `0.90710x`

The isolated savings do not add linearly to the graph timing because the full
autograd graph has allocation, scheduling, and surrounding linear backward work.

## 3. Raw kernel versus integration overhead

The raw backward candidate measured approximately `56–71 us` in the saved-P
boundary evaluator, versus `177–249 us` for the analytical PyTorch boundary;
geometric mean speedup was `3.4578x`. The raw forward candidate was about
`56–61 us` for the larger shapes, versus `61–90 us` in the component harness.

The integrated autograd Function performs Python dispatch, output allocation,
`ctypes` launch, saved-tensor registration, and later custom backward dispatch.
The backward library lookup was initially repeated per backward call; this was
fixed by caching the loaded library. The clean rerun improved the training
result from the earlier `0.8268x` measurement to `0.9071x`, proving that
repeated extension lookup was a real integration artifact. It did not remove
the complete regression.

## 4. Allocation and saved-tensor audit

The integrated forward allocates one `[S,B,I/TP]` FP16 output. The custom
Function saves the full packed FC1 intermediate `[S,B,2I/TP]`; at the largest
shape this is `128*2*8192*2 = 4,194,304` bytes, excluding allocator metadata.
The optimized backward allocates a full packed grad-intermediate of the same
logical size. The candidate kernel itself has no workspace or temporary tensor.

The original path uses PyTorch autograd nodes for SiLU, multiply, and split and
therefore also saves activation inputs/outputs and creates multiple temporary
buffers. The exact internal saved-node representation is version-dependent;
the custom path makes the 4 MiB packed intermediate explicit. No duplicate
copy or forced contiguous conversion was observed in the integration source.

This explicit save is a likely contributor to graph-level overhead, but the
available measurements do not isolate its exact microsecond cost independently
of autograd scheduling.

## 5. NSYS comparison

The original training trace contains the same FC1/FC2 Volta FP16 Tensor Core
GEMMs plus separate PyTorch SiLU, sigmoid, multiply, add, split/cat, and
backward elementwise kernels. The integrated trace contains the same GEMM
families plus `swiglu_kernel` for forward and `k(const __half*, const __half*,
__half*, int, int, __half)` for backward.

The Phase 14-F training NSYS command was:

`nsys profile --trace=cuda,nvtx,osrt --sample=none -o phase14d_train python integration_harness.py --mode train`

The integrated trace showed the optimized backward kernel with 10 instances
and approximately `0.608 ms` aggregate in the profiled repetitions, alongside
the optimized forward kernel and unchanged GEMMs. CUDA API activity was
dominated by kernel launches; no new global synchronization or memcpy pattern
was introduced by the adapter.

## 6. Quantitative attribution

Measured facts support four contributors:

1. GEMM dominance: isolated FC1+FC2 forward already cost about `557 us` at the
   largest shape, before either activation direction or linear backward.
2. PyTorch elementwise graph cost: reference activation forward/backward was
   about `383.9 us` in the isolated component run, and NSYS showed many kernels.
3. Integration overhead: custom autograd, output/grad allocation, saved packed
   intermediate, and Python/ctypes dispatch remain on the integrated path.
4. Extension lookup: repeated backward library lookup was measured as an
   integration defect; caching it changed the training result from `0.8268x` to
   `0.9071x`.

No evidence supports extra synchronization, hidden `.contiguous()` copies, or
repeated compilation as the remaining cause. The residual `0.9071x` is therefore
attributed to graph/allocation/autograd overhead plus unchanged surrounding
backward GEMMs, not to an incorrect candidate kernel.

## 7. Amdahl limits

Using the largest-shape isolated reference activation costs (`89.5 us` forward
and `294.4 us` backward) over the original complete training time (`2133.6
us`) gives approximate fractions of `4.2%` and `13.8%`. The ideal limits are:

- infinite-speed forward activation: `1/(1-0.042) = 1.044x`
- infinite-speed backward activation: `1/(1-0.138) = 1.160x`
- both infinitely fast: `1/(1-0.180) = 1.219x`

These are upper bounds for this measurement decomposition, not observed gains.
The observed integrated result below 1x proves integration overhead is larger
than the saved activation time in this harness. Further activation-only work
has limited headroom; reducing the FC1 output materialization is a more
promising boundary if implementation complexity is acceptable.

## 8. Plumbing fix and remeasurement

Only one integration plumbing fix was made: cache the loaded backward shared
library instead of calling `ctypes.CDLL` for every custom backward invocation.
No mathematical kernel was changed and no candidate was regenerated.

After the fix: forward correctness, gradient correctness, Phase 14-B/C/D
regressions, and upstream integrity remained PASS. The clean full training
measurement is `0.9071x`; historical Phase 14-D/E measurements remain
preserved.

## 9. FC1 runtime characterization

Megatron source calls `linear_fc1` at `MLP.forward` line 264, then the activation
branch, then `linear_fc2` at line 346. The Phase 14 harness uses an FP16 local
linear adapter equivalent to `F.linear(x, weight)`, with TP=1 and no bias. NSYS
shows Volta FP16 `s884gemm` Tensor Core kernels for these GEMMs. FC1 writes its
complete packed output to global memory before the activation kernel reads it;
the current NSYS graph contains a separate activation launch.

The FC1 interface is `[S,B,H] -> [S,B,2I/TP]`, contiguous row-major FP16,
with gate in the first half and up in the second half. Official bias is null;
the general source branch can pass bias, but it is outside the measured
configuration.

## 10. Fusion feasibility

An FC1-plus-SwiGLU epilogue is technically feasible only through a custom GEMM
epilogue, CUTLASS/WMMA implementation, or a library facility that can express
gated SiLU. Ordinary cuBLAS GEMM does not provide this gated nonlinear epilogue
by itself. V100 FP16 Tensor Core GEMM is supported, but FP8/modern TE paths are
excluded. The research boundary is recorded in
`targets/megatron_5be9626/swiglu/fc1_epilogue_candidate_boundary.json` and is
not active.

## 11. Traffic model

For `N=S*B` and `W=2I/TP`, the current FC1/SwiGLU interface writes `N*W*2`
bytes for the FP16 FC1 output and then reads `N*W*2` bytes in SwiGLU. The
activation output writes `N*(W/2)*2` bytes and remains necessary. With no bias,
an FC1 epilogue could theoretically eliminate `2*N*W*2 = 4*N*W` bytes of
intermediate write/read traffic:

- `[16,1,1024]`: `0.5 MiB`
- `[64,2,1024]`: `4.0 MiB`
- `[128,2,1024]`: `8.0 MiB`

These are analytical traffic savings, not measured speedups. A fused backward
design may need to save/recompute different values and can give back some of
this benefit.

## 12. Backward implications

Forward fusion can avoid writing the packed FC1 preactivation, but backward
needs gate/up values for the derivative and FC1 weight gradient. Options are to
save the preactivation, save a transformed representation, or recompute FC1.
Recomputation changes the training boundary and may erase forward memory
savings. The FC1 weight-gradient path also requires the original hidden states
and the gradient entering FC1. These questions must be resolved before any
active fusion contract is created.

## 13. Engineering decision

The current activation-only boundary has verified forward `2.83x`, backward
`3.46x`, MLP-forward `1.023x`, and full-training `0.9071x`. The data movement
and GEMM measurements show that an FC1 epilogue covers more meaningful work,
but it requires a custom V100 GEMM implementation and a new backward storage or
recompute design. Therefore FC1+SwiGLU is the evidence-supported next research
direction, not an implementation started in Phase 14-F.

## Integrity and limitations

No new Agent campaign or CUDA candidate was generated. Existing forward and
backward incumbents were not modified. The exact Megatron checkout remains at
`5be9626709af2722333bf54797c954c09edeada3` with a clean working tree.

Remaining unknowns include exact allocator-level saved-tensor costs under a
production Megatron configuration, cuBLASLt/CUTLASS epilogue expressiveness on
the installed V100 toolchain, and the optimal backward strategy for a fused
FC1 boundary.
