# Phase 20-A — Vendor-GEMM-Preserving Attention Qualification

## Inherited evidence and strategy change

Phase 17 qualified real native dense/no-mask/p=0 FP16
`DotProductAttention`. A2’s one full-core custom kernel was good only at S=16
(5.928x), near-neutral at S=64 (1.167x), and harmful at S=128 (0.347x); A3
then failed S=128 correctness. Phase 20-A therefore excludes full-core
row-serial work and preserves vendor QK/PV GEMMs. Inherited evidence is frozen
in `phase20a_inherited_evidence.json`.

## Real graph, GEMM, and scale audit

The source is `DotProductAttention.forward`: global score workspace is acquired
as `[B*heads,S,S]` FP16, then `torch.baddbmm(... beta=0.0,
alpha=self.softmax_scale)` computes QK. Here `self.softmax_scale=0.125`, so
scale is already folded into the vendor GEMM alpha. There is no separate scale
kernel to remove. Q/K are views/transposes of `[S,B,heads,64]`; PV is vendor
`torch.bmm` after probabilities are viewed `[B*heads,S,S]`. Trace kernel
families establish Volta FP16 GEMM execution, but do not justify a more
specific unsupported cuBLAS implementation claim.

`masked_softmax_fusion=false` makes `FusedScaleMaskSoftmax` take its Torch
fallback. It converts FP16 scores to FP32, calls last-axis softmax, and casts
probabilities to FP16. In this configuration its internal `scale` is `None`,
because QK alpha already supplied 0.125; no mask is applied and dropout is
identity. This yields softmax warp plus conversion/copy kernels.

## Workspace and kernel attribution

Score workspace is a preallocated global-memory-buffer FP16 tensor; allocation
is outside the original timed boundary. Scores/probabilities require
`B*16*S*S` elements. Score FP16 / FP32 softmax transient / FP16 probabilities
are respectively 8/16/8 KiB at S=16, 256/512/256 KiB at S=64, and
1024/2048/1024 KiB at S=128. These sizes describe materialization, not an
allocation-cost claim.

| shape | total GPU kernels us | QK+PV us | softmax us | conversion/copy/local us | ideal no-non-GEMM bound |
|---|---:|---:|---:|---:|---:|
| 16×1 | 76.577 | 31.041 | 8.896 | 36.640 | 2.467x |
| 64×2 | 100.705 | 45.728 | 13.984 | 40.993 | 2.203x |
| 128×2 | 139.168 | 58.240 | 25.024 | 55.904 | 2.389x |

Copy/conversion-only elimination bounds are only 1.112x, 1.097x, and 1.104x.
All figures are kernel-time upper bounds, not event-scope predictions or
candidate scores. The non-GEMM graph is a meaningful but bounded research
opportunity; it must retain a real FP32 softmax.

## Evaluator-only hybrid prototype

`phase20a_hybrid_validation.py` is explicitly
`EVALUATOR_VALIDATION_ONLY_NOT_AGENT_CANDIDATE_NOT_BENCHMARK_ELIGIBLE`. It runs
vendor `baddbmm` with alpha 0.125, explicit FP32 softmax/cast, and vendor `bmm`.
It matched native Megatron exactly (`max_abs_reference=0`) for all official
shapes; independent FP32-oracle errors after FP16 output cast were 0.000977,
0.001465, and 0.001465. It launches no collective.

Its separate microbench timings are attribution only and are not summed into an
attention score. Diagnostic S=64 NSYS showed the expected two Volta FP16 GEMMs,
softmax warp, and local conversion/copy graph: six launches for both real
reference and hybrid, 0 NCCL. The prototype proves feasibility and also shows
that merely re-expressing the same PyTorch operations does not reduce the graph.

## Current-stream, workspace, and future ABI

Future work must retain framework-owned orchestration: native/PyTorch vendor
GEMMs execute on the current PyTorch stream; the local replacement receives
the score/probability tensor and stream. This avoids a candidate-owned cuBLAS
handle, hidden default-stream work, or handle construction inside timing.
Equivalent score/output workspace is preallocated. Persistent workspace is
only fair if both paths have equivalent availability or a required temporary is
semantically eliminated. Required output completion remains in scope.

Candidate classes, if later authorized, are `VENDOR_GEMM_PLUS_CUSTOM_SOFTMAX`,
`VENDOR_GEMM_PLUS_FUSED_SCALE_SOFTMAX`, and
`VENDOR_GEMM_PLUS_CUSTOM_LOCAL_GRAPH`. `FULL_CORE_ROW_SERIAL` is excluded.

## Statistical policy and environment readiness

Phase 20-A creates no score. A later Phase 20-B may reuse
`native_dot_product_attention_stability_v1` only after proving exact Phase 17
scope equivalence; otherwise it must freeze a new policy before candidates.

The read-only readiness probe at S=64 had CV 0.1545, but the state was
`NOISY` rather than clean: GPU0 was 76 C/P0 with 258 MHz SM clock; GPU1 had an
unrelated CUDA Python process and 100% utilization. This is provenance only,
not a baseline or a reason to delete later samples.

## Verdict, authorization, CLI, and integrity

Verdict: `QUALIFIED_FOR_VENDOR_GEMM_AGENT_CAMPAIGN`. The qualification rests
on a real, non-GEMM graph of 45–81 us and an exact hybrid feasibility proof,
not a claimed speedup. `phase20b_authorization.json` is metadata only and does
not authorize an Agent episode in this phase. Existing reproducible target
commands remain `python -m lab.cli target --target
megatron_native_dot_product_attention --action {inspect,replay,benchmark,profile}`.

Knowledge is isolated under the attention target namespace. CE forward/backward,
RMSNorm, Phase 17 attention state, and MoE states remain unchanged. Megatron
HEAD is `5be9626709af2722333bf54797c954c09edeada3` and its working tree is
clean. Remaining limitations are no custom local kernel validation yet, noisy
current environment, and no claim that a local fusion can attain the upper
bounds.
