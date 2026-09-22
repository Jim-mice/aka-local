# Real Optimization Loop 002 — Physical Boundary Measurement

## Environment

The workload was the already validated local authentic Megatron MLP path, not the nine-grid workload:

- Python: `C:\Users\38154\.venvs\urban6-stgcn\Scripts\python.exe`
- PyTorch: `2.11.0+cu128`; CUDA runtime `12.8`
- GPU: local `NVIDIA GeForce RTX 5060 Laptop GPU`, device 0
- Megatron commit: `5be9626709af2722333bf54797c954c09edeada3`
- TP=1, SP=false, batch=2, sequence=3, hidden=8, FFN hidden=16, float32
- Path: `MLP.forward -> bias_swiglu_impl -> BiasSwiGLUFunction -> linear_fc2`

Raw environment evidence is in `artifacts/integration/swiglu/real_loop_002/measurement_environment.json`.

## Profiler capability

- Nsight Systems: `UNAVAILABLE` (`where.exe nsys` found no executable).
- Nsight Compute: `PASS`, version `2026.3.0.0`; metric query and NVTX-filtered profiles completed.
- `torch.profiler` CUDA activities: `BLOCKED` by `CUPTI_ERROR_INVALID_DEVICE`. The context returned, but Kineto reported that CUDA profiler activities were missing.

The raw capability record and probe output are in `profiler_capabilities.json` and `torch_profiler_probe.txt`.

## Measurement harness

`scripts/measure_swiglu_boundary_002.py` adds process-local NVTX ranges around the real FC1, authentic `bias_swiglu_impl`, and FC2 calls. It does not replace the algorithm or modify Megatron. The harness completed 20 warmups after the first compile invocation and 120 CUDA Event samples.

## Kernel attribution

Nsight Compute was run separately with NVTX filters `AKA_FC1]`, `AKA_SWIGLU]`, and `AKA_FC2]`. Each range yielded one unique kernel execution in the one-iteration workload:

| Range | Kernel count | Evidence |
|---|---:|---|
| FC1 | 1 | NVTX-filtered NCU |
| SWIGLU | 1 | `triton_poi_fused_add_mul_silu_split_0` |
| FC2 | 1 | cuBLAS GEMV kernel |

Therefore `SEPARATE_SWIGLU_FC2_KERNELS = YES` for this workload. NCU replay IDs are not counted as extra application executions.

## Materialization hierarchy

- `GRAPH_VALUE_EXISTS = YES`: source and runtime call path.
- `DEVICE_TENSOR_MATERIALIZED = SUPPORTED`: SwiGLU returns a device tensor and FC2 is a separate kernel consumer.
- `DRAM_ROUND_TRIP = UNKNOWN`: kernel DRAM counters cannot isolate activation traffic from weights and other traffic.

The last statement is intentionally conservative: separate kernels do not, by themselves, prove an HBM round trip.

## Nsight Compute evidence

For the filtered authentic SwiGLU kernel, current-version metrics reported approximately:

- kernel duration: `3648 ns`
- registers/thread: `16`
- achieved occupancy: `7.25%`
- dynamic/static shared memory per block: `0 / 0 bytes`
- total kernel DRAM read: `8448 bytes`
- total kernel DRAM write: `0 bytes`
- total kernel L2 request bytes: recorded in the raw metric artifact

These are kernel-total measurements, not activation-only measurements. The raw CSV files are retained under `artifacts/integration/swiglu/real_loop_002/`.

## CUDA Event evidence

The 120-sample instrumented run produced:

- whole MLP median: `0.263312 ms`, CV `0.0470`
- authentic SwiGLU boundary median: `0.072512 ms`, CV `0.1304`
- FC2 median: `0.073008 ms`, CV `0.0336`
- SwiGLU+FC2 measured-region median: available in `boundary_timing.json`

The separate uninstrumented 120-sample run produced a whole MLP median of `0.191952 ms`. The inner Event instrumentation therefore changed the median by approximately `+37.18%`.

The boundary timing is useful for scale and ordering, but it must not be treated as a non-perturbing model fraction.

## Measurement perturbation

`EVENT_INSTRUMENTATION = PERTURBING`.

The uninstrumented comparison is retained in `artifacts/integration/swiglu/real_loop_002/uninstrumented/boundary_timing.json`. No paired instrumented fraction is used as an unperturbed claim.

## PerformanceFacts v6

`performance_facts_v6.json` distinguishes `MEASURED`, `SOURCE_DERIVED`, `DERIVED`, and `UNKNOWN`. In particular:

- measured: separate kernel execution, device tensor boundary, kernel-total duration/resource metrics, CUDA Event distributions;
- source-derived: authentic bias+SwiGLU path and FC2 consumer relationship;
- unknown: activation-only DRAM write/read, DRAM round trip, exact launch API overhead, backward kernel attribution, activation-vs-weight traffic.

No UNKNOWN field was converted to zero.

## Structural hypothesis revisit

`fp-producer-consumer-boundary` is now `PARTIAL`:

1. physical kernel separation is measured;
2. a device intermediate is supported;
3. kernel cost/resource facts exist;
4. avoidable activation traffic and a beneficial transformation are not established.

It is not yet valid to implement fusion or claim materialization elimination.

## System-value gate

Using the prior authentic local minimal GPTModel split fraction `f = 0.0155713429`:

| Hypothetical SwiGLU speedup | Local model upper bound |
|---:|---:|
| 1.25x | 1.003124x |
| 1.5x | 1.005218x |
| 2x | 1.007847x |
| 4x | 1.011817x |
| infinite | 1.015818x |

`SYSTEM_VALUE_GATE = SCIENTIFIC_ONLY`: the evidence is useful for causal validation, but the current local model does not justify pretending a large system-throughput opportunity.

## Candidate decision

`CANDIDATE_DECISION = NOT_READY`. No optimization candidate was created, and no L0/L1/L2 candidate evaluation was run.

## Knowledge delta

`knowledge_delta_measurement.json` proposes only an `INCONCLUSIVE` delta. It is not written to Mechanism Memory. Confirmed observations are limited to the distinction between graph adjacency and physical kernel separation/device materialization; activation DRAM round-trip remains inconclusive.

## Next action

Before implementing `fp-producer-consumer-boundary`, obtain representative-shape evidence that separates activation traffic from FC2 weight traffic and establishes whether the boundary is avoidable without changing forward/backward semantics. Until then, do not create a candidate.

## Final status

```text
MEASUREMENT_EPISODE = PASS
NSYS_STATUS = UNAVAILABLE
NCU_STATUS = PASS
TORCH_PROFILER_STATUS = BLOCKED
FC1_KERNEL_COUNT = 1
SWIGLU_KERNEL_COUNT = 1
FC2_KERNEL_COUNT = 1
SEPARATE_SWIGLU_FC2_KERNELS = YES
DEVICE_TENSOR_MATERIALIZED = YES
ACTIVATION_DRAM_ROUND_TRIP = UNKNOWN
SWIGLU_REGISTERS_PER_THREAD = 16
SWIGLU_ACHIEVED_OCCUPANCY = 7.25%
SWIGLU_DRAM_BYTES = 8448 (kernel-total read, not activation-only)
EVENT_INSTRUMENTATION = PERTURBING
PERFORMANCE_FACTS_V6 = PASS
STRUCTURAL_PRECONDITION = PARTIAL
SYSTEM_VALUE_GATE = SCIENTIFIC_ONLY
CANDIDATE_DECISION = NOT_READY
OPTIMIZATION_CANDIDATE_CREATED = NO
KNOWLEDGE_DELTA = INCONCLUSIVE
MECHANISM_MEMORY_MODIFIED = NO
NINE_GRID_E2E_EXECUTED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
```
