# Phase 15-C.3 — CE Backward Stability Recovery

Status: ENVIRONMENT/STABILITY BLOCKED. This is a complete independent requalification under the unchanged Phase 15-C.1/15-C.2 scientific policy; no backward incumbent was created.

## Frozen starting state

The forward campaign remains `FORWARD_CAMPAIGN_FROZEN`; its Episode 7 artifact was not changed. The backward replay, performance, statistical, and active optimization contract hashes remain respectively `4841f90840fee98eeff354e999fd25a1626c99e561cd5402b0dc68ec8f0d94da`, `afbfcff56ee6891b888ca7c6fbf632b74ccfab71f849d7934c56dc1d4bd9e205`, `7a750bc7f4f3c7b768bd5b35d695dab6762131d4c6681d6925100bf5557cf18f`, and `6121f49401f3ef4601549c8732a62870ff17a7a7d7047e9c55d55b79c9ff6e92`.

B1 source hash remained `008155d13eda72ab1feb267a06d360a13ad2730128a4aca4cfe8a3af70e374ab`. Rebuild provenance used the verified remote command `nvcc -O3 -arch=sm_70 -shared -Xcompiler -fPIC`; the historical rejected `-std=c++17` and `--std=c++17` records remain preserved.

## Original B1 result preserved

The original B1 raw campaign is now explicitly marked `STABILITY_BLOCKED_ORIGINAL_RUN`. Its 2.55717x raw geometric mean remains historical/provisional only.

The 1010.688 us reference outlier was rank 0 / physical GPU0, invocation 1, config 32x2x64, ABBA ordinal 3 (final reference), block 1 iteration 4. The 717.856 us reference outlier was rank 0 / physical GPU0, invocation 2, config 8x1x32, ABBA ordinal 3, block 2 iteration 7. Both rows are valid and remain in the original data. Rank 1 did not show a matching excursion; the raw data therefore supports physical-GPU or scheduling variability, but cannot identify a hardware root cause.

## Diagnostics and environment provenance

A reference-only stage diagnostic ran before requalification. It observed local `prepare_gradient_calculation_operands` and `calculate_gradients` work with zero collective calls. Its purpose was attribution only, not scoring. The qualified boundary retains zero NCCL.

The fixed mapping was `CUDA_VISIBLE_DEVICES=0,1`, rank 0 -> GPU0 PCI `00000000:08:00.0`, rank 1 -> GPU1 PCI `00000000:84:00.0`. System state before and after the new run is preserved in the v2 raw artifact. No unrelated process was killed and no clock, power, persistence, driver, or system setting was modified.

## Independent v2 campaign

`episode_B1/stability_requalification_v2/` is a new complete campaign: three independent invocations, three blocks of ten samples, five warmups, and A-B-B-A order for every official TP=2 configuration. It uses the same B1 source, fixture identity, physical mapping, reference, out-of-place ABI, timing boundary, and frozen CV <= 0.20 gate. It contains 180 raw samples per implementation/rank/config because each A-B-B-A invocation has two occurrences of each implementation. No old raw row is included.

| Config | Reference rank0 CV | Candidate rank0 CV | Reference proxy us | Candidate proxy us | Raw ratio |
|---|---:|---:|---:|---:|---:|
| 8x1x32 | 0.08729 | 0.15961 | 286.874 | 116.697 | 2.4583x |
| 32x2x64 | 0.17065 | 0.20803 | 306.836 | 120.230 | 2.5521x |
| 128x2x128 | 0.13855 | 0.10842 | 311.103 | 114.154 | 2.7253x |

All reference ranks qualify. Candidate rank 0 at 32x2x64 fails the frozen gate (0.20803 > 0.20). Its valid maximum was 412.672 us and was retained. Therefore the terminal verdict is `STABILITY_BLOCKED_CANDIDATE`. The raw 2.57620x geometric mean is not a promotion score and no bootstrap CI is reported.

## Decision

The new campaign does not support promotion, even though the direction and rough magnitude are consistent with the original provisional B1 result. Old and new scores were not averaged. No incumbent, deterministic replay, candidate promotion NSYS, B2/B3 benchmark, B4/B5 episode, integration, or combined-training result was created.

Diagnosis remains `MULTI_KERNEL_OVERHEAD`, `LOCAL_KERNEL_LATENCY`, and `ENVIRONMENTAL_VARIABILITY`; no memory-bound or compute-bound claim is made. Generic framework lesson: a fast candidate cannot be promoted when either side of its predeclared paired stability gate fails.

## Integrity

Megatron HEAD was verified as `5be9626709af2722333bf54797c954c09edeada3`; its working tree was clean. Forward artifacts and their frozen collective structure were not modified.
