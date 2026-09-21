# Phase 20-B.0 — Attention Scoring-Era Qualification

## Outcome

`ENVIRONMENT_NOT_READY_FOR_SCORING`. No Phase 20-B V1 candidate, active
baseline, score, or promotion artifact was created.

## Era and scope

The new era is `attention_vendor_gemm_hybrid_v1`. Its semantic replay is the
unchanged Phase 17 dense/no-mask/p=0 FP16 local core. Scope is
`PERFORMANCE_SCOPE_REUSED`: preallocated Q/K/V, vendor QK with alpha=0.125,
FP32 softmax, FP16 probabilities, vendor PV, and output completion; QKV,
rotary, output projection, allocation, backward, and unrelated synchronization
are excluded. Phase 17 means are historical and are not a Phase 20 baseline.

## Event versus kernel timing reconciliation

The native S=16/S=64/S=128 diagnostic event means were 867.0/863.6/898.0 us;
synchronized wall means were 878.9/871.1/878.1 us. Batched 16-call event
means were 783.5/799.0/800.0 us per call, so batching removes only roughly
64–98 us per call. Empty host `cuda.synchronize` cost was about 47–48 us.
Events were created once per shape, recorded on the current stream, and reused;
event creation is excluded.

In contrast, inherited NSYS GPU kernel totals are 76.577/100.705/139.168 us.
Fresh exact-source manual staging reproduced real output exactly but reported
roughly 1.01–1.08 ms event totals; even the no-kernel global-buffer-lookup to
next-event interval was about 94–104 us. Thus the gap is not explained by
Python loop/event construction alone. It is consistent with stream scheduling/
queue delay and current environmental contention, but is not assigned to a
candidate-addressable local kernel.

| shape | vendor QK+PV kernel us | local GPU graph us | event-scope us | all-local-kernel elimination bound within event scope |
|---|---:|---:|---:|---:|
| 16×1 | 31.041 | 45.536 | 867.010 | 1.055x |
| 64×2 | 45.728 | 54.977 | 863.602 | 1.068x |
| 128×2 | 58.240 | 80.928 | 898.015 | 1.099x |

These scoring-scope bounds are upper bounds only. Copy/conversion-only work is
still smaller; vendor GEMM time, Python/dispatcher work, timing mechanics, and
unattributed scheduling delay are not local-stage candidate targets.

## Candidate-addressable boundary

The frozen future contract preserves framework-owned vendor QK and PV. The
only candidate ABI is score FP16 to probability FP16:
`attention_dense_softmax_fp16_stream(scores, probabilities, batch_heads,
sequence_length, stream)`. Scores are flattened contiguous `[B*heads,S,S]`,
read-only, and already scaled by alpha=0.125. The candidate must do FP32
row-softmax and write separate FP16 probabilities; no Q/K/V, mask, scale,
dropout state, or cuBLAS handle is exposed. Framework-owned orchestration is
selected to avoid hidden default-stream/handle behavior.

The future correctness harness is ready: identical vendor QK/PV surrounds
reference or candidate local stages; final output is compared with real
Megatron and independent FP32 oracle, and probabilities with the local FP32
softmax oracle. Edge and negative fixtures are frozen in the harness metadata.

## Environment and policy

The one Phase 20-B.0 read-only readiness probe had CV 0.0516 but is classified
`NOISY`: GPU0 was 76–77 C at 258 MHz SM clock, while GPU1 had an unrelated CUDA
Python process at 100% utilization. This is not scoring evidence and does not
permit sample deletion. The existing attention ABBA/CV policy may be reused
only if this identical scope is later used; CV ≤0.20 and raw retention remain
immutable.

Because environment readiness is not `READY`, no active reference baseline was
run, no Phase 20 baseline manifest exists, and V1 is not authorized. A fresh
explicitly authorized clean/READY-window readiness assessment is the next legal
action; it must then repeat all gating before any baseline or candidate work.

## Integrity

CE forward/backward, RMSNorm, Phase 17 attention, and MoE remain frozen. No
candidate was generated. Megatron HEAD is
`5be9626709af2722333bf54797c954c09edeada3` and the working tree is clean.
