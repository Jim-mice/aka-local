# Phase 17-B.2 — Attention Profile-Guided Closure

## A2 profile

A2 source hash is preserved in episode_A2. REAL NSYS was captured for S=16/B=1, S=64/B=2, and S=128/B=2 using the exact local core boundary. Reference produced six kernel instances per shape and zero NCCL. A2 produced one custom kernel per shape and zero NCCL. A2 GPU kernel totals were approximately 37.3 us, 609.7 us, and 2258.0 us; reference totals were 76.6 us, 100.7 us, and 139.2 us.

The trace supports `FULL_CORE_KERNEL_SCALING` and `SERIAL_WORK_GROWTH`: A2 manually performs full QK/softmax/PV work in one kernel, while native execution retains vendor GEMMs. This explains the small-S launch/fusion benefit and the severe S=128 regression without making unsupported occupancy or memory claims.

## A3

A3 was justified by the concrete S=128 profile. It was a new real Agent episode, not a manual A2 edit. Preflight and CUDA 11.8/sm70 build passed. S=16 and S=64 correctness passed, but S=128 failed against both real reference and oracle with max absolute error `0.3759765625` and max relative oracle error `90.2064`, exceeding the frozen `0.005` tolerance. A3 is `REJECT_CORRECTNESS`; it was not benchmarked or profiled.

## Terminal decision

Attention state: `AGGREGATE_WIN_PER_CONFIG_MIXED` with no promoted incumbent. A2 remains the best valid candidate evidence (geometric mean 1.338883x, CI95 [1.307531x, 1.368066x]) but the S=128 regression prevents an unconditional promotion claim and the project has no rule authorizing a mixed severe regression as incumbent. No A4 is justified: A3 exposed no valid new runtime strategy, only a correctness blocker. A1/A2/A3 artifacts and raw data remain preserved.
