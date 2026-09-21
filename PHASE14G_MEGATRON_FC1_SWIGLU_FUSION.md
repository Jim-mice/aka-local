# Phase 14-G — FC1 + SwiGLU fusion feasibility

## Verdict

Phase 14-G is **BLOCKED BEFORE AGENT CAMPAIGN**. The exact runtime has cuBLAS
and cuBLASLt, but CUDA 11.8 cuBLASLt exposes standard/ReLU/GELU-style epilogues,
not the required two-half gated SiLU epilogue. CUTLASS was not found in the
audited remote paths. CUDA WMMA is available, but no validated fused GEMM
implementation exists in aka-local; writing one would itself be a new custom
GEMM optimization campaign. No Agent candidates were generated and no two-kernel
fallback was mislabeled as fusion.

## 1. Motivation and exact real boundary

Phase 14-F measured activation-only forward/backward gains of approximately
`2.83x` and `3.46x`, but complete MLP forward+backward remained `0.9071x`.
The exact Megatron source calls `linear_fc1` at `megatron/core/transformer/mlp.py`
line 264, then the activation branch, then `linear_fc2` at line 346.

The measured TP=1 boundary is:

`hidden_states [S,B,H] FP16 -> FC1 -> packed [S,B,2I/TP] -> gate/up split -> SiLU(gate)*(up+offset)`.

Official configuration has `H=1024`, `I=4096`, no FC1 bias, contiguous row-major
FP16 tensors, gate in the first half and up in the second half, and offset zero.
FC2, communication, dropout/BDA, TE, and FP8 are excluded.

## 2. Toolchain audit

Remote evidence was collected on V100 sm70 with `/usr/local/cuda-11.8/bin/nvcc`
11.8.89:

- cuBLAS: available.
- cuBLASLt header and shared library: available.
- CUTLASS: no checkout/package found under audited `/usr/local` and
  `/home/<REMOTE_USER>` paths.
- CUDA WMMA headers: available through the CUDA toolkit.
- Transformer Engine/Apex: unavailable in the verified runtime.

The CUDA 11.8 `cublasLt.h` epilogue enumeration contains DEFAULT, RELU,
RELU_AUX, BIAS, GELU auxiliary and backward-related variants. It contains no
SiLU epilogue and no operation that takes the first half of an accumulator as a
gate and the second half as an up/value vector before multiplying them.

The full machine-readable audit is
`targets/megatron_5be9626/swiglu/fc1_swiglu_backend_audit.json`.

## 3. Backend decision

cuBLAS/cuBLASLt are available but insufficient. CUTLASS is unavailable as an
existing dependency. WMMA/custom tiled GEMM is possible in principle, but would
require a new V100 GEMM implementation with tile scheduling, FP32 accumulation
matching, epilogue register management, resource tuning, and numerical
validation. It would also require a new backward saved-state design even though
backward is outside this phase.

That is not an existing practical backend to hand to an Agent. Starting that
implementation would violate the requested pre-campaign feasibility gate and
would turn this phase into a new GEMM optimization campaign. Therefore work
stopped before creating an active FC1-SwiGLU contract or candidates.

## 4. Baseline/runtime evidence recovered

Phase 14-F already established via REAL NSYS that the original graph launches a
Volta FP16 Tensor Core FC1 GEMM, writes the complete packed FC1 output to global
memory, and then launches separate activation kernels. The current analytical
traffic model predicts removal of the packed output write plus reread:

- `[16,1,1024]`: `0.5 MiB`
- `[64,2,1024]`: `4.0 MiB`
- `[128,2,1024]`: `8.0 MiB`

These remain theoretical traffic savings, not a fused-candidate measurement.
No expanded-boundary baseline or candidate NSYS was fabricated because there is
no valid fused backend.

## 5. Candidate ABI/status

No active `fc1_swiglu_optimization_contract.json` was created. This is
intentional: the implementation backend gate failed. A future ABI must consume
hidden states, FC1 weights, and any real bias inputs directly; accepting the
materialized packed FC1 output would defeat the fusion objective.

The research-only boundary remains in
`fc1_epilogue_candidate_boundary.json`, with status
`research_only_not_active`.

## 6. Backward implications

Forward fusion cannot simply discard all FC1 preactivation state: SwiGLU
backward needs gate/up values, and FC1 weight gradients need the appropriate
FC1 input and gradient path. A future design must choose between saving packed
preactivation, saving a transformed representation, or recomputing FC1. This
may reduce or erase the forward traffic benefit.

## 7. Why no Agent campaign was started

There are no episode manifests, benchmarks, incumbent, replay, or candidate
NSYS artifacts for `fc1_swiglu/`. Creating a two-launch candidate would violate
the explicit requirement that separate FC1 and activation kernels not be called
fusion. Creating a naive custom GEMM candidate would bypass the required backend
feasibility audit and would not be defensible as a small reconnaissance phase.

## 8. Knowledge and integrity

No forward activation, backward activation, or standalone operator knowledge was
modified. No fused incumbent exists. The authoritative Megatron checkout was
rechecked at:

`HEAD = 5be9626709af2722333bf54797c954c09edeada3`

and `git status --short` was empty.

## 9. Next engineering prerequisite

Before Phase 14-G can resume, provide or intentionally build one of:

1. a vendored CUTLASS version with a demonstrably usable custom epilogue;
2. a reviewed WMMA/custom GEMM implementation for sm70 with FP16 inputs and
   correct FP32 accumulation; or
3. another installed library that explicitly supports gated SiLU over packed
   FC1 output.

Only after that prerequisite should the expanded contract, trusted reference,
baseline, Agent campaign, true-fusion NSYS proof, and optional sidecar
integration be created.

## Final acceptance matrix

| Requirement | Status |
|---|---|
| Real FC1 boundary recovered | PASS |
| V100 backend audited | PASS |
| Existing practical fused backend verified | FAIL/BLOCKER |
| Active fusion contract | NOT STARTED BY DESIGN |
| Agent candidates | NOT STARTED BY DESIGN |
| Two-kernel fallback mislabeled fusion | NO |
| Megatron upstream modified | NO |
| Prior evidence preserved | PASS |

Phase 14-G is therefore a scientifically valid blocker result, not a fusion
success. No kernel optimization was started.
