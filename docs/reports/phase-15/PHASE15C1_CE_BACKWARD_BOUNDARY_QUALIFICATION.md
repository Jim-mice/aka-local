# Phase 15-C.1 — Megatron Vocab-Parallel CE Backward Boundary Qualification

Status: QUALIFIED_WITH_ENVIRONMENT_CAVEAT — no backward Agent campaign was started.

This phase qualifies a new rank-local backward boundary only. The Phase 15-B forward campaign remains frozen as FORWARD_CAMPAIGN_FROZEN; its Episode 7 incumbent and all forward score history were not changed.

## 1. Frozen Phase 15-B.8 provenance

The frozen forward artifacts bind to Megatron commit 5be9626709af2722333bf54797c954c09edeada3, semantic contract 112959ca63020e9b, evaluator tp2_semantic_v2 / 7cc0fac8d80b2d27, fixture coherent_global_fixture_v2, performance era tp2_comparable_v1, and forward collective structure 1 MAX + 2 SUM. The forward campaign state remains FORWARD_CAMPAIGN_FROZEN.

Episode 7 remains the forward incumbent at stable geometric-mean TP=2 speedup 2.4771907400291515x; no backward number is combined with that score.

## 2. Exact Megatron backward source path

The required checkout was re-read at the required commit. The authoritative path is:

LanguageModule.compute_language_model_loss (megatron/core/models/common/language_module/language_module.py:155-200) -> unfused tensor_parallel.vocab_parallel_cross_entropy (cross_entropy.py:213-235) -> _VocabParallelCrossEntropy.forward (cross_entropy.py:119-183) -> ctx.save_for_backward -> loss.backward() -> _VocabParallelCrossEntropy.backward (cross_entropy.py:185-210) -> VocabParallelCrossEntropy.prepare_gradient_calculation_operands (cross_entropy.py:81-98) -> VocabParallelCrossEntropy.calculate_gradients (cross_entropy.py:100-116).

This qualification uses the native unfused path with label_smoothing=0.0. Transformer Engine and the fused native CE implementation are excluded from this boundary.

The line-level map is preserved in targets/megatron_5be9626/vocab_parallel_cross_entropy/backward_source_mapping.json.

## 3. Exact backward equations

Let N=S*B, L=V/TP, p[r,j] be the saved FP32 local softmax, m[r] be the saved target mask, and t[r] be the saved safe local target index.

Megatron first creates:

~~~text
grad_2d = softmax.view(N, L)       # alias, not a copy
arange_1d = arange(0, N)           # created by the prepare helper
softmax_update[r] = 1 - float(m[r])
~~~

For the no-smoothing path it then performs, in place:

~~~text
grad_2d[r, t[r]] -= softmax_update[r]
grad_input[r, :] *= grad_output[r]
~~~

On the owning rank, m[r]=false, so the target probability is subtracted. On a non-owning rank, m[r]=true and t[r]=0, so the safe index is accessed but the subtraction weight is zero. There is no backward all-reduce.

## 4. Saved tensors and ownership semantics

The real forward saves exactly exp_logits after in-place normalization, target_mask, and masked_target_1d at cross_entropy.py:178-181. The replay harness obtains these from real ctx.saved_tensors; it does not reconstruct a simplified state for timing.

Observed saved-state types and layout:

| Value | Shape | Dtype | Layout / behavior |
|---|---:|---|---|
| saved softmax | [S,B,V/TP] | FP32 | contiguous; aliases the gradient buffer and is mutated in place |
| target mask | [S,B] | bool | contiguous; true for non-owning targets |
| masked target | [N] | int64 | contiguous; non-owning entries are safely zero |
| grad output | [S,B] | FP32 | already available upstream input |
| direct gradient | [S,B,V/TP] | FP32 | same storage as saved softmax |

The public .grad attached to the original FP16 logits is FP16 because autograd casts the direct FP32 return to the input dtype after the qualified boundary. That cast is outside the direct local backward timing boundary.

## 5. Backward replay contract

Created targets/megatron_5be9626/vocab_parallel_cross_entropy/backward_replay_contract.json.

Contract hash: 4841f90840fee98eeff354e999fd25a1626c99e561cd5402b0dc68ec8f0d94da.

The contract freezes the real saved-state roles, FP32 direct boundary, aliasing, target ownership, zero collectives, official configurations, stream semantics, and tolerances. It explicitly excludes label smoothing from this no-smoothing boundary.

## 6. Isolated replay harness

remote_backward_replay.py generates a coherent global FP16 fixture, runs real Megatron forward outside timing, clones the actual saved tensors, and then exercises the exact source backward helpers. The timed region never reruns forward or constructs saved state.

Per measurement, synchronization is performed before the CUDA event interval and after event completion. prepare_gradient_calculation_operands and calculate_gradients execute on the current stream between AKA_CE_BACKWARD_BEGIN / AKA_CE_BACKWARD_END.

The output allocation and saved-state restore copy are outside timing. The exact helper-created arange_1d and softmax_update temporaries remain inside timing because they are part of the current Megatron implementation.

## 7. Correctness results

TP=1 and REAL TP=2 official configurations all passed. Every TP=2 rank independently passed direct-gradient comparison against the full-vocabulary FP32 oracle. The maximum observed direct FP32 gradient error was 2.98e-8; saved-softmax maximum error was 2.98e-8.

The public FP16 input-gradient comparison showed the expected cast quantization, with maximum absolute error approximately 4.80e-4, far below the inherited FP16 gradient tolerance 0.03.

Correctness evidence is preserved in targets/megatron_5be9626/vocab_parallel_cross_entropy/backward_correctness_raw.json, including rank-local summaries and hashes.

## 8. Finite-difference and autograd sanity

Independent FP64 finite difference on a two-row, five-class diagnostic gave maximum absolute error 3.48984813269837e-10 with epsilon 1e-6.

The exact Megatron custom autograd path also passed. Direct source-helper gradients and public autograd gradients differ only by the expected FP32-to-FP16 input-gradient cast.

## 9. Target ownership and edge fixtures

TP=2 edge fixtures covered:

- target owned by rank 0 at the first local index;
- target owned by rank 1 at the last local index;
- non-owning rank safe local index;
- equal logits;
- large positive and negative logits;
- global maximum on either rank.

For all cases, real target mask, masked local target, saved softmax, and direct gradient semantics matched the independent oracle on both ranks. No invalid local target indexing was used for a non-owning rank.

## 10. Zero-collective proof

The source backward implementation contains no torch.distributed call. The only distributed operations are in forward, before ctx.save_for_backward.

Runtime NSYS was gated with cudaProfilerStart after forward and warmup. In the gated capture, cuda_gpu_kern_sum contains only local ATen/PyTorch kernels and no ncclDevKernel_* entry. nvtx_pushpop_sum contains only AKA_CE_BACKWARD_BEGIN; no NCCL range occurs inside the backward range. Therefore the qualified backward boundary has exactly zero collectives.

The capture command and raw summaries are in targets/megatron_5be9626/vocab_parallel_cross_entropy/backward_nsys_run.txt, backward_nsys_stats_rank0.txt, and backward_nsys_stats_rank1.txt.

## 11. NVTX/range provenance

The sidecar harness emits fixed ranges:

~~~text
AKA_CE_BACKWARD_BEGIN
  CUDA event start
  exact Megatron prepare helper
  exact Megatron calculate helper
  CUDA event end
AKA_CE_BACKWARD_END
~~~

The NSYS capture uses --capture-range=cudaProfilerApi --stop-on-exit=true; after capture starts, the profiling run performs no distributed barriers, so NCCL cannot be hidden in a captured coordination barrier.

## 12. Backward performance contract

Created targets/megatron_5be9626/vocab_parallel_cross_entropy/backward_performance_contract.json.

Performance contract hash: afbfcff56ee6891b888ca7c6fbf632b74ccfab71f849d7934c56dc1d4bd9e205.

The boundary is saved_state_to_local_logits_gradient. It excludes forward, all three forward collectives, saved-state construction, fixture generation, process startup, output allocation, barriers, event completion synchronization, autograd bookkeeping, and the FP16 cast-back. It includes the exact helper-created arange_1d and softmax_update temporaries.

## 13. Scoring rule for a future campaign

Because no collective exists inside this boundary, rank 0 and rank 1 are not collapsed into a per-sample distributed completion time. Every rank is reported independently.

For future training-critical-path comparison, the frozen conservative per-configuration proxy is:

~~~text
proxy_latency(config) = max(mean_rank0(config), mean_rank1(config))
~~~

This is explicitly a critical-path proxy, not a measured synchronization or communication cost. A future candidate must be compared against the exact same rank-aware rule and same physical mapping.

## 14. Statistical policy and baseline era

Created targets/megatron_5be9626/vocab_parallel_cross_entropy/backward_statistical_policy.json.

Statistical policy hash: 7a750bc7f4f3c7b768bd5b35d695dab6762131d4c6681d6925100bf5557cf18f.

New era: tp2_ce_backward_local_v1.

Policy:

- three independent invocations;
- three blocks per invocation;
- ten measurements per block/config/rank;
- five warmups per invocation/config;
- 90 raw samples per rank/config;
- arithmetic mean is primary; median/MAD are diagnostics only;
- slow valid samples are retained;
- missing/failed/mismatched rows are invalid and fail closed;
- fixed maximum rank CV qualification limit is 0.20;
- future A/B uncertainty uses paired block bootstrap, seed 1515, 10,000 resamples.

No forward statistical policy or benchmark era was reused.

## 15. Stable local baseline

Created targets/megatron_5be9626/vocab_parallel_cross_entropy/backward_baseline_manifest.json. Raw runs are preserved as backward_benchmark_raw_run1.json, backward_benchmark_raw_run2.json, and backward_benchmark_raw_run3.json; the combined analysis is backward_baseline_statistics.json.

| Config | Rank 0 mean / CV | Rank 1 mean / CV | Critical-path proxy |
|---|---:|---:|---:|
| S=8,B=1,V=32 | 289.757 us / 0.1609 | 47.468 us / 0.0261 | 289.757 us |
| S=32,B=2,V=64 | 284.603 us / 0.1013 | 48.879 us / 0.0166 | 284.603 us |
| S=128,B=2,V=128 | 299.485 us / 0.1283 | 60.541 us / 0.0168 | 299.485 us |

All 90 samples per rank/config are complete. The policy threshold is met, so the baseline is qualified for future A/B measurement, with an explicit environmental caveat: the host was not exclusive. GPU1 had an unrelated CUDA Python process and GPU0 had higher temperature/clock variability. No process was killed and no clock, power, persistence, driver, or system setting was changed.

## 16. Stage-level backward graph

The exact source path expands into eight reported GPU kernel families in the gated NSYS kernel report:

1. arange_cuda_out index kernel;
2. target-index index_elementwise_kernel;
3. target-index index_put_kernel;
4. FP32 direct-copy / saved-softmax copy;
5. fill kernel(s) for temporary initialization;
6. FP32 multiply kernel for grad_output scaling;
7. add kernels for tensor iterator arithmetic;
8. second elementwise indexed/update family.

Across five profiled iterations the kernel report shows the corresponding local instances and no NCCL kernel. The API report shows launch, event, memcpy, and synchronization calls, but those host/API timings are profiling diagnostics and are not substituted for the CUDA-event baseline score.

## 17. Allocation and materialization audit

| Object | Boundary behavior |
|---|---|
| real forward saved state | generated outside backward timing |
| softmax template | cloned outside timing |
| target mask | cloned outside timing |
| masked target | cloned outside timing |
| grad output | preallocated outside timing |
| gradient output storage | aliases preallocated softmax buffer; no output allocation inside timing |
| arange_1d | created by exact prepare_gradient_calculation_operands inside timing |
| softmax_update | created by exact prepare_gradient_calculation_operands inside timing |
| NCCL buffers | none in backward timing |

The main future headroom is therefore local kernel launch/materialization and indexed update structure, not communication removal.

## 18. Future replacement boundary

Created research-only targets/megatron_5be9626/vocab_parallel_cross_entropy/backward_candidate_boundary.json, hash 2e8b7fa616fcd1c67cdd011458e8946fb983c7f566bd871743fac06080f61545.

The proposed ABI is intentionally not active:

~~~cpp
extern "C" void ce_backward_local_fp32_stream(
    const float* softmax,
    const bool* target_mask,
    const int64_t* masked_target_1d,
    const float* grad_output,
    float* grad_input,
    int64_t rows,
    int64_t local_vocab,
    cudaStream_t stream);
~~~

The ABI has explicit semantic roles, accepts the real saved-state representation, permits the reference alias grad_input == softmax, and has no process-group or NCCL argument because the boundary is rank-local. Whether arange_1d and softmax_update should remain internal or become explicit inputs is a future contract decision, not an optimization decision in this phase.

## 19. Golden feasibility

The exact Megatron helper path itself is the correctness golden for this qualification. A separate CUDA golden candidate was not created because the existing Python source path already passed both TP=1 and real TP=2 under the exact evaluator, and no candidate ABI is active yet. No golden implementation is eligible for performance scoring or incumbent promotion.

## 20. Optimization headroom

Backward-only baseline is approximately 285–299 us on the slower physical GPU for the official TP=2 configurations under the captured environment. It executes multiple small local kernels and creates two helper temporaries inside the boundary. The trace supports a MULTI_KERNEL_OVERHEAD / LOCAL_KERNEL_LATENCY research opportunity, but it does not establish memory-bound, compute-bound, or Tensor Core behavior.

The old 0.87 ms / 5.32 ms values are not reused: they included rank waiting from the old distributed stage diagnostic.

## 21. Relationship to frozen forward incumbent

Episode 7 remains untouched and remains the forward incumbent. The backward baseline is a different boundary and different benchmark era. A future full CE training estimate may conceptually add optimized forward plus unoptimized backward, but no combined score is published here because the frozen forward boundary and this isolated backward boundary have different orchestration and timing contracts.

The new evidence says backward local work is a real remaining local opportunity, while forward local optimization has already reached diminishing returns against the mandatory 1 MAX + 2 SUM boundary. Whether backward optimization is worthwhile should be decided in Phase 15-C.2 using this contract.

## 22. CLI

The stable target CLI now supports:

~~~text
python lab/cli.py target --target vocab_parallel_cross_entropy --action inspect-backward
python lab/cli.py target --target vocab_parallel_cross_entropy --action replay-backward
python lab/cli.py target --target vocab_parallel_cross_entropy --action benchmark-backward
python lab/cli.py target --target vocab_parallel_cross_entropy --action profile-backward
~~~

These actions are sidecar/harness actions. They do not patch Megatron or start an Agent campaign.

## 23. Knowledge namespace

Backward-only evidence is isolated under knowledge/targets/megatron_5be9626/vocab_parallel_cross_entropy/backward/knowledge_summary.json. Forward Episode 7 scores and standalone operator knowledge were not modified.

## 24. No Agent campaign

No Episode B1 or backward Agent candidate was generated. There is no backward incumbent and no backward speedup claim. The next allowed action is a separately approved Phase 15-C.2 campaign using the frozen backward replay/performance contracts.

## 25. Upstream integrity

Final verification:

~~~text
Megatron HEAD: 5be9626709af2722333bf54797c954c09edeada3
Megatron working tree: clean
~~~

No authoritative Megatron file was edited.

## 26. Remaining limitations

1. The host had an unrelated CUDA process and nonuniform physical-GPU state; the baseline is qualified with this caveat, not a claim of exclusive-hardware reproducibility.
2. The direct boundary returns FP32 while the public FP16 input gradient is cast by autograd outside the boundary; a future candidate must preserve this distinction.
3. The NSYS profile is a local kernel graph diagnostic, not an NCU hardware bottleneck analysis.
4. No backward candidate, speedup, incumbent, or full-training improvement exists yet.
5. Label smoothing and Transformer Engine/fused CE paths are outside this Phase 15-C.1 contract.
