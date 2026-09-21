# Phase 15-B.5 Benchmark Scope Equivalence

## Verdict

The Phase 15-A baseline is scientifically valid for its own original Megatron replay, but it is not scope-equivalent to the repaired Phase 15-B candidate harness. The old hash `f11fd1fc6846d6f2` is therefore preserved as historical evidence and is not used for a new candidate speedup claim.

## 1. Recovered Phase 15-B.4 state

Episodes 7, 8, and 10 now produce complete two-rank records under evaluator `tp2_semantic_v2` (`7cc0fac8d80b2d27`). The row collector fails closed and aggregates the slower rank per iteration. This resolves the missing-row defect, but complete rows do not by themselves prove timing equivalence.

## 2. Baseline timing sequence

The Phase 15-A `replay.py --bench` path allocates a fresh `z=local.detach().clone().requires_grad_()` inside every measured interval, calls the real Megatron `vocab_parallel_cross_entropy`, calls `y.sum().backward()`, records the end event, synchronizes, and reads elapsed time. Its measured sequence is:

`barrier → start event → clone/input allocation → Megatron CE forward (local compute + MAX + SUM target + SUM denominator + loss/saved state) → autograd backward/local gradient → end event → CUDA synchronize → elapsed`.

The baseline also performs ten warmup iterations, with the same clone/autograd pattern.

## 3. Candidate timing sequence

The repaired candidate path allocates `lm`, `pred`, `den`, `ex`, `mask`, and `tl` before the event. Its measured sequence is:

`barrier → pre-event CUDA synchronize → start event → local_max ABI → MAX all-reduce → local_prepare ABI → SUM target → SUM denominator → explicit softmax from ex/den → manual rank-local gradient arithmetic → end event → CUDA synchronize → elapsed`.

It does not allocate a fresh differentiable local input or invoke PyTorch autograd inside the measured interval. Its five warmup iterations also use the preallocated-stage pattern.

## 4. Stage equivalence matrix

| Stage | Phase 15-A baseline | Candidate harness | Verdict |
|---|---|---|---|
| target fixture | coherent global fixture | coherent global fixture | equivalent |
| target-mask storage | produced by Megatron implementation | candidate ABI buffer, sized `[rows, local_vocab]` | representation differs |
| local max | Megatron/PyTorch path | `local_max_fp16_stream` | semantic equivalent, implementation differs |
| MAX all-reduce | inside event | inside event | equivalent |
| target extraction | Megatron implementation | `local_prepare_fp16_stream` | semantic equivalent only after correctness |
| SUM target | inside event | inside event | equivalent |
| exp/denominator | Megatron implementation | `local_prepare` plus SUM | semantic equivalent only after correctness |
| SUM denominator | inside event | inside event | equivalent |
| softmax saved state | Megatron autograd saved state | explicit `ex/den` | not proven allocation/state-equivalent |
| loss finalization | Megatron CE output and backward source | not independently finalized as Megatron autograd output | not equivalent |
| backward | PyTorch autograd `y.sum().backward()` | manual local gradient arithmetic | not equivalent |
| input/output allocation | clone and autograd allocation inside timing | workspaces outside timing | not equivalent |

The decisive mismatches are allocation scope, autograd/backward implementation, and loss/saved-state production. Consequently the current candidate numbers cannot be divided by the Phase 15-A means as scientific speedups.

## 5. Allocation and synchronization audit

Baseline-only timed allocation: differentiable local clone and autograd graph/gradient storage. Candidate-only outside-timed allocation: local reduction outputs, exponent buffer, mask, and target-local buffer. Both paths synchronize before elapsed time is queried, but they synchronize after different semantic work. Both include the three real collectives. Current-stream ordering is valid for the candidate ABI, but stream validity does not repair the scope mismatch.

## 6. Configuration identity

The logical configurations match: TP=2, FP16, coherent fixture version, and token/vocabulary products `[8,32]`, `[64,64]`, `[256,128]`, represented by Megatron shapes `[8,1,32]`, `[32,2,64]`, and `[128,2,128]`. This is necessary but not sufficient for score comparability.

## 7. Performance contract and benchmark era

`targets/megatron_5be9626/vocab_parallel_cross_entropy/performance_contract.json` freezes the required future scope: preallocated outputs/workspaces outside timing, the same stage sequence, three mandatory collectives, completed CUDA events, and max-rank aggregation. Its status is `AUDIT_SCOPE_NOT_EQUIVALENT`; its hash is recorded after creation. The old baseline is marked `SCOPE_NOT_EQUIVALENT` for this candidate harness. A new baseline era must be measured with a shared/aligned reference and candidate orchestrator before any speedup is published.

## 8. Required aligned harness

The next safe implementation is one distributed timing driver with interchangeable local implementations. It must use the same fixtures, allocations, barrier, event placement, loss/saved-softmax outputs, local gradient boundary, collective calls, and rank aggregation for both the trusted reference and candidate. Only the local implementation symbols may differ.

## 9. Correctness and historical integrity

Episodes 7/8/10 remain corrected-evaluator valid; Episodes 6/9 remain incorrect and are excluded. Phase 15-A through 15-B.4 artifacts were not overwritten. No incumbent, NSYS promotion profile, or profile-guided Agent episode is created by this phase because no scope-equivalent score exists.

## 10. Final status

`SCOPE_NOT_EQUIVALENT`. The old baseline is not reused, and no speedup is claimed. The next required work is a new comparable baseline under `tp2_scope_v1`, followed by candidate reruns, NSYS, diagnosis, and promotion review.

Megatron integrity remains required at commit `5be9626709af2722333bf54797c954c09edeada3` with a clean working tree.
