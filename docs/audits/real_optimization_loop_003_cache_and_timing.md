# Real Optimization Loop 003 — Cache Residency and Timing Attribution

## Exact activation footprint

Frozen authentic local workload: batch=2, sequence=3, hidden=8, FFN hidden=16, float32, TP=1, SP=false.

- FC1 output: `[3, 2, 32]`, 192 elements, 768 B payload.
- SwiGLU activation handed to FC2: `[3, 2, 16]`, 96 elements, contiguous, stride `[32,16,1]`, 384 B payload/storage.
- Bias: 128 B.
- GPU-reported L2 cache: 32 MiB; this was read from `torch.cuda.get_device_properties(0).L2_cache_size`.

The 384 B activation payload is not equivalent to kernel DRAM traffic.

## Timing hierarchy

Measurements used the authentic `bias_swiglu_impl` without replacing its algorithm:

| Layer | Median |
|---|---:|
| Python wall around synchronized call | 65.100 µs |
| Single outer CUDA Event around call | 73.872 µs |
| Batched outer Event, K=512 amortized | 53.746 µs/invocation |
| Previous inner Event measurement | 72.512 µs |
| NCU active SwiGLU kernel duration | 3.648 µs |

Ratios to active NCU kernel duration are approximately 17.85x, 20.25x, 14.73x, and 19.88x respectively for the current Python wall, single Event, compliant batched Event, and previous inner Event. The NCU value is active kernel execution only. The outer CUDA Event includes GPU timeline gaps and all queued work in the repeated region; the Python wall also includes host synchronization and launch/API overhead. None of these should be called “kernel duration”.

## Batched amortized timing

The final compliant batched protocol used 20 warmup batches and 50 measured batches, with K=512 authentic invocations per batch. The median measured window was approximately 27.518 ms, giving 53.746 µs per invocation. The per-invocation CV was approximately 1.37%; p10/p90 were approximately 52.35/54.37 µs. An earlier K=8192 trial produced a 456 ms window and was retained only as diagnostic evidence for choosing K; it is not the final estimate.

This reduced the single-call Event boundary effect but did not collapse to 3.648 µs, showing that the repeated authentic callable has substantial launch/framework/stream-gap cost beyond active kernel time. It does not identify which component is dominant.

## Launch geometry

NCU reported the authentic SwiGLU kernel as:

- grid: `(1,1,1)`;
- block: `(128,1,1)`;
- threads/block: 128;
- derived warps/block: 4;
- blocks: 1;
- device SM count: 26;
- registers/thread: 16;
- dynamic/static shared memory: 0/0 B;
- achieved occupancy: 7.25%;
- NCU theoretical-occupancy guidance: 100% for this tiny launch.

`LOW_OCCUPANCY_CAUSE = GRID_UNDERSUBSCRIBED`. The evidence does not support attributing the low achieved occupancy to register pressure or shared-memory limits.

## Occupancy diagnosis

One 128-thread block cannot occupy the full 26-SM device. This is a tiny-shape launch fact, not a general statement about production SwiGLU. The local shape is therefore unsuitable for choosing a large-model kernel strategy.

## Cache-pressure design

The cache-pressure run used the same authentic activation values, the same FC2 weights, and the same FC2 shape. The pressure buffer was 64 MiB, twice the device-reported 32 MiB L2 size, and was touched before FC2. It was a measurement perturbation only and did not change the numerical output contract.

## FC2 baseline vs pressure

NCU used metrics queried from the installed NCU 2026.3.0:

| Metric | Baseline | Pressure |
|---|---:|---:|
| FC2 duration | 6.368 µs | 6.304 µs |
| FC2 total DRAM read | 25088 B | 24832 B |
| FC2 total L2 request bytes | 82912 B | 82912 B |

These are FC2 total-kernel metrics and include weights/other traffic. They are not activation-only bytes. The pressure comparison showed no stable increase, so it provides no positive evidence that FC2 is strongly dependent on activation cache residency in this local workload.

## Memory hierarchy interpretation

The evidence chain remains deliberately separated:

- `GRAPH_VALUE_EXISTS = YES`;
- `DEVICE_TENSOR_MATERIALIZED = YES`;
- `SEPARATE_KERNELS = YES`;
- `CACHE_RESIDENCY = INCONCLUSIVE` (no reliable cache-hit metric and no pressure response);
- `DRAM_ROUND_TRIP = UNKNOWN` as a direct fact.

For the specific causal story “there is an avoidable activation HBM round trip between SwiGLU and FC2”, the local evidence is negative: 384 B activation, zero SwiGLU kernel DRAM writes in the prior NCU run, no FC2 pressure response, and a one-block tiny launch. Therefore `DRAM_ROUND_TRIP_CAUSAL_STORY = REFUTED_FOR_LOCAL_WORKLOAD`. This does not prove that no DRAM traffic exists in any larger shape.

## Structural hypothesis verdict

`fp-producer-consumer-boundary = REJECTED` for the frozen local workload. Separate kernels and a device tensor are real, but they do not establish a worthwhile or even present HBM-round-trip elimination opportunity here.

## Local-shape representativeness

`LOCAL_SHAPE_PERF_GENERALIZATION = UNSAFE`. The shape is one block on a 26-SM GPU, cache pressure is ineffective as a discriminator, and the minimal-model operator fraction is only about 1.56%. Correctness and integration evidence remain valid; performance conclusions must wait for representative real model shapes.

## Project next action

`NEXT_PROJECT_ACTION = SWIGLU_LOCAL_HYPOTHESIS_REJECTED`.

Stop SwiGLU fusion/materialization optimization for this local shape and wait for a real nine-grid/representative Megatron workload. Do not create a candidate from this evidence.

## Knowledge delta

`knowledge_delta_003.json` is proposed only and is not written to Mechanism Memory. It records confirmed timing/geometry observations, refutes the HBM-elimination causal story for this local workload, and leaves activation-only cache/DRAM attribution and nine-grid generalization inconclusive.

## Final status

```text
MEASUREMENT_EPISODE_003 = PASS
ACTIVATION_BYTES = 384
NCU_SWIGLU_KERNEL_US = 3.648
BATCHED_SWIGLU_US = 53.746
OLD_INNER_EVENT_SWIGLU_US = 72.512
OLD_EVENT_OVERHEAD_CONFIRMED = YES
SWIGLU_GRID_BLOCKS = 1
SWIGLU_THREADS_PER_BLOCK = 128
LOW_OCCUPANCY_CAUSE = GRID_UNDERSUBSCRIBED
FC2_BASELINE_DRAM_READ = 25088
FC2_PRESSURE_DRAM_READ = 24832
FC2_BASELINE_L2_READ = 82912
FC2_PRESSURE_L2_READ = 82912
ACTIVATION_HANDOFF = INCONCLUSIVE
DRAM_ROUND_TRIP_CAUSAL_STORY = REFUTED_FOR_LOCAL_WORKLOAD
LOCAL_SHAPE_PERF_GENERALIZATION = UNSAFE
STRUCTURAL_HYPOTHESIS = REJECTED
NEXT_PROJECT_ACTION = SWIGLU_LOCAL_HYPOTHESIS_REJECTED
OPTIMIZATION_CANDIDATE_CREATED = NO
KNOWLEDGE_DELTA = REFUTED
MECHANISM_MEMORY_MODIFIED = NO
NINE_GRID_E2E_EXECUTED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
```


## Timing reconciliation correction

The authoritative value is `53.74621972441673 µs` from the final raw K=512 artifact. The prior `55.63945323228836 µs` value came from an earlier K=8192 diagnostic batch and is retained only as provenance, not as the final measurement.
