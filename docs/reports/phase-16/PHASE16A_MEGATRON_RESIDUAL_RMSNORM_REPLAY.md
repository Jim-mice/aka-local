# Phase 16-A — Megatron residual / RMSNorm replay qualification

## Verdict

`Residual Add RMSNorm` was an ambiguous historical label, not one unconditional Megatron operator.  The qualified replay target is **Megatron non-TE Torch RMSNorm** (`WrappedTorchNorm` -> `torch.nn.RMSNorm`).  It is `QUALIFIED_FOR_AGENT_CAMPAIGN` as a local forward boundary only.  The TE residual-fork path is `DEPENDENCY_BLOCKED`; BDA is an independent, RNG-sensitive boundary and was not silently included in RMSNorm.

The CE forward campaign remains `FORWARD_CAMPAIGN_FROZEN`; CE backward remains `ENVIRONMENT_STABILITY_BLOCKED`.  No CE artifacts or Megatron source were changed, and no Phase 16 Agent candidate was generated.

## Source map and execution graph

| Concrete boundary | Exact source | Semantics | V100 status |
|---|---|---|---|
| BDA | `megatron/core/fusions/fused_bias_dropout.py:11` | `residual + dropout(x + optional_bias)` | REAL_REPLAY_AVAILABLE, but training RNG must be replayed exactly |
| Torch RMSNorm | `megatron/core/transformer/torch_norm.py:27,58-63` | `WrappedTorchNorm` creates `torch.nn.RMSNorm` | REAL_REPLAY_AVAILABLE |
| TE fused residual RMSNorm | `megatron/core/extensions/transformer_engine.py:626-793` | `MakeExtraOutput` residual fork plus TE RMSNorm; returns `(normalized_output, residual)` | DEPENDENCY_BLOCKED |

`TransformerLayer` invokes input normalization at `transformer_layer.py:621-635`, handles the TE tuple result there, then invokes BDA after branch results around lines 676-687 and 972-983.  The real order is:

`hidden_states -> input normalization -> attention/MLP branch -> (branch output,bias) + residual through BDA -> merged hidden state -> next normalization`.

Thus BDA occurs adjacent to normalizations in a Transformer block but is not part of the non-TE RMSNorm callable.  The machine-readable map and graph are [source_map.json](C:/Users/38154/projects/aka-local/targets/megatron_5be9626/residual_rmsnorm/source_map.json) and [execution_graph.json](C:/Users/38154/projects/aka-local/targets/megatron_5be9626/residual_rmsnorm/execution_graph.json).

## Exact semantics and dependencies

For the selected path, with last dimension `H`, the replayed equation is:

`y = cast_fp16(x_fp32 * rsqrt(mean(x_fp32^2, -1) + eps) * weight_fp32)`.

There is no bias, no residual merge, no dropout/RNG, and no collective in this local boundary.  RMSNorm output is FP16; the independent oracle accumulates in FP32.  PyTorch autograd supplies gradients for `x` and `weight`.

BDA accepts optional bias, conditionally upcasts `x` and bias to residual dtype, uses in-place operations only in eval with no gradients, invokes `torch.nn.functional.dropout`, and adds residual.  Its training mode requires the exact dropout Philox/RNG consumption.  A separately sampled `torch.rand_like` mask did not reproduce that stream; this was recorded as an **unqualified oracle attempt**, not a BDA source failure.  Its eval (`p=0.1`, no dropout) comparison differed by at most one FP16 ULP (`0.00390625`).

The actual remote environment is PyTorch `2.7.1+cu118` with CUDA available and `torch.nn.RMSNorm` present.  `import transformer_engine` raises `ModuleNotFoundError`; Megatron itself reported Transformer Engine/Apex fallback warnings.  Consequently `TEFusedResidualRMSNorm` was not replayed and was not replaced by a Torch approximation.

## Replay, correctness, and numerical checks

Qualification fixtures are the explicitly assumed representative FP16 V100 family `(S,B,H) = (16,1,1024), (64,2,1024), (128,2,1024)`.  They are not claimed to come from a particular model checkpoint.

Real Megatron construction used `TransformerConfig(normalization='RMSNorm')` and `WrappedTorchNorm`; the remote replay evidence is [phase16a_replay_raw.json](C:/Users/38154/projects/aka-local/targets/megatron_5be9626/residual_rmsnorm/phase16a_replay_raw.json).

- Forward oracle: maximum absolute and relative error was `0` on all three fixtures.
- Backward against the independent FP32 equation: worst observed `x` gradient max-abs was `0.00132179`; worst `weight` gradient max-abs was `0.00387478`.  Both are within the declared FP16-facing `0.005` tolerance and all gradients were finite.
- A tiny FP64 finite-difference check gave maximum absolute error `2.57474e-11`.
- RMSNorm has no RNG state.  BDA training RNG is separately documented rather than conflated with this contract.

## Contracts and future boundary

The replay contract is [replay_contract.json](C:/Users/38154/projects/aka-local/targets/megatron_5be9626/residual_rmsnorm/replay_contract.json), hash `5588960649248c710c9ae57f0b613ee4e881564aa8bf6a4361b2f56276bcecb6`.

The qualification performance boundary is preallocated FP16 contiguous `x` and `weight` on the current CUDA stream through the real `WrappedTorchNorm` output.  Construction, allocation, fixtures, backward, and startup are excluded.  CUDA events are recorded on the current stream and the end event completes before elapsed-time retrieval.  This is frozen in [performance_contract.json](C:/Users/38154/projects/aka-local/targets/megatron_5be9626/residual_rmsnorm/performance_contract.json), hash `86a7faf7a7cb02c0e4a2fbf8599e0f60d353079e607cb8c2900668e3d79d95b6`.

The research-only future boundary is [candidate_boundary.json](C:/Users/38154/projects/aka-local/targets/megatron_5be9626/residual_rmsnorm/candidate_boundary.json).  It deliberately exposes semantic inputs (`x`, weight, epsilon) rather than PyTorch implementation temporaries.  No ABI or optimization contract is active and no candidate exists.

## Baseline and real NSYS

Thirty event-timed samples per configuration produced these raw baseline means:

| Shape | Mean us | Std us |
|---|---:|---:|
| 16x1x1024 | 236.750 | 20.720 |
| 64x2x1024 | 233.846 | 22.584 |
| 128x2x1024 | 238.115 | 18.515 |

These are qualification measurements, not an Agent-campaign score or a statistical promotion baseline.

The real NSYS command was:

```text
/usr/local/cuda-11.8/bin/nsys profile --force-overwrite=true --trace=cuda,nvtx --capture-range=cudaProfilerApi --stop-on-exit=true -o /tmp/aka_phase16a_rmsnorm <REMOTE_HOME>/venvs/lerobot-act/bin/python /tmp/aka_phase16a.py --mode profile --S 64 --B 2 --H 1024
```

It captured the `AKA_RMSNORM_BEGIN..END` replay region after setup.  For 20 real `WrappedTorchNorm` calls, NSYS reported 160 CUDA launches: eight ATen kernel families per call (mean reduction, elementwise operations, direct copy, add, power, rsqrt, and FP16 conversion).  Total reported kernel time was about `2.296 ms` over 20 calls; `cudaLaunchKernel` accounted for 160 API calls with 15.034 us average API duration.  No NCCL kernel appears in the captured CUDA kernel summary.  Full trace summary and provenance are in [phase16a_nsys_summary.json](C:/Users/38154/projects/aka-local/targets/megatron_5be9626/residual_rmsnorm/phase16a_nsys_summary.json).

## Diagnosis and headroom

Evidence supports `MULTI_KERNEL_OVERHEAD`, `LOCAL_KERNEL_LATENCY`, and launch overhead for the Torch fallback.  It does not establish memory-bound, compute-bound, occupancy, or register conclusions.  Eight local kernels per call at roughly 234–238 us is enough measured overhead to justify a later bounded local CUDA investigation, provided a new phase freezes a candidate ABI, a statistically qualified baseline policy, and source-compatible saved/backward semantics.

TE residual-fork + RMSNorm is still dependency-blocked, not an optimization target on this host.  BDA is a separate potential target; its training replay must first freeze exact dropout RNG/mask semantics and therefore is not combined with this RMSNorm verdict.

## CLI and isolation

The stable target interface is:

```text
.\.venv\Scripts\python.exe -m lab.cli target --target residual_rmsnorm --action inspect
.\.venv\Scripts\python.exe -m lab.cli target --target residual_rmsnorm --action replay
.\.venv\Scripts\python.exe -m lab.cli target --target residual_rmsnorm --action benchmark
.\.venv\Scripts\python.exe -m lab.cli target --target residual_rmsnorm --action profile
```

Knowledge is isolated at [knowledge_summary.json](C:/Users/38154/projects/aka-local/knowledge/targets/megatron_5be9626/residual_rmsnorm/knowledge_summary.json).  No SwiGLU or CE knowledge/score files were changed.

## Integrity and limitations

Megatron was rechecked at `5be9626709af2722333bf54797c954c09edeada3` with a clean working tree.  Limitations: no TE package is available, BDA training mask replay is not yet qualified, shapes are stated representative assumptions rather than checkpoint-derived, and Phase 16-A intentionally created no Agent candidates or active optimization ABI.
