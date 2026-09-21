# Phase 15-B — Megatron Vocab-Parallel CE Agent Campaign

## Verdict

**MECHANICS PARTIAL / OPTIMIZATION BLOCKED.** The distributed optimization contract and real TP=2 baseline were frozen, five real Codex Agent episodes were generated, and all candidates were contained without modifying Megatron. No candidate reached the required contract/build/TP=1/TP=2 correctness chain, so no performance claim, incumbent, candidate NSYS profile, or deterministic winning replay is reported.

## 1. Distributed stage graph

The exact source is `megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy` at commit `5be9626709af2722333bf54797c954c09edeada3`.

| Stage | Inputs | Local outputs | Collective/barrier | Saved state |
|---|---|---|---|---|
| local max | rank-local FP16 logits `[N,V/TP]` | FP32 max `[N]` | MAX all-reduce | none |
| shift/extract/exp | logits, global max, global targets | local predicted logit `[N]`, local exp denominator `[N]`, exp/softmax storage, mask, masked target | two SUM all-reduces: predicted then denominator | exp/softmax, target mask, masked target |
| loss/normalize | global predicted and denominator, local exp | loss `[N]`, normalized softmax | none | normalized softmax |
| backward | saved tensors and grad output | rank-local gradient | none | n/a |

The production baseline preserves the three barriers: `MAX`, `SUM(predicted target logit)`, `SUM(exp denominator)`.

## 2. Optimization contract

`targets/megatron_5be9626/vocab_parallel_cross_entropy/optimization_contract.json`

Contract hash: `112959ca63020e9b`.

Only rank-local CUDA computation is replaceable. Process-group initialization, NCCL collectives, global vocabulary semantics, and backward optimization are outside this phase. Candidate code receives local FP16 logits, targets, and the global max produced by the real MAX all-reduce; it returns local contributions for the real SUM all-reduces.

## 3. Collective-coalescing feasibility

The two SUM operands are both available before the source issues the two SUM calls, so packing them into one `[N,2]` FP32 SUM all-reduce is algebraically **VALID as a research variant**, subject to numerical and saved-state verification. It is not active in the baseline contract or candidates. The campaign therefore preserves exactly three collectives and does not claim coalescing.

## 4. TP=2 baseline

The primary score is rank-synchronized TP=2 forward-plus-backward latency including all three real NCCL collectives. Startup, imports, model-parallel initialization, fixture generation, and setup are excluded. Baseline hash: `f11fd1fc6846d6f2`.

| shape `[S,B,V]` | mean us | std us | CV |
|---|---:|---:|---:|
| `[8,1,32]` | 2774.02 | 41.42 | 0.0149 |
| `[32,2,64]` | 3138.56 | 388.01 | 0.1236 |
| `[128,2,128]` | 2812.93 | 20.33 | 0.0072 |

Rank-1 measurements remain in `tp2_bench.txt`; promotion would require both ranks to pass correctness and benchmark.

## 5. Agent episodes

| Episode | Agent artifact | Result | Reason |
|---:|---|---|---|
| 1 | candidate generated | REJECT_COMPILE | remote nvcc reported `CUDART_INF_F` undefined |
| 2 | candidate generated | REJECT_COMPILE | same portable-constant failure |
| 3 | candidate generated | REJECT_COMPILE | same portable-constant failure |
| 4 | candidate generated | REJECT_CONTRACT | built, but required `local_max_fp16_stream` ABI symbol was missing |
| 5 | candidate generated | REJECT_CONTRACT | same ABI symbol failure |

All five episodes contain `candidate.cu`, `hypothesis.json`, `AGENT.md`, `agent_prompt.txt`, `result.json`, `decision.json`, and `episode_manifest.json`. No candidate was manually repaired.

## 6. Profile feedback loop

Episodes 2–5 prompts included:

```text
=== CURRENT TP=2 PERFORMANCE ===
=== LOCAL COMPUTE PROFILE ===
=== NCCL PROFILE EVIDENCE ===
=== CURRENT DIAGNOSIS ===
=== PROFILE-GUIDED RECOMMENDATIONS ===
=== WHAT WORKED ===
=== WHAT FAILED ===
=== NEXT RECOMMENDED SEARCH ===
```

The prompts carried Phase 15-A real NSYS evidence: PyTorch reduction/index kernels and `ncclDevKernel_AllReduce_Sum_f32_RING_LL`, with the instruction to preserve all three collective barriers. No candidate became valid enough for candidate-specific NSYS feedback.

## 7. TP=1 and TP=2 correctness

The trusted Phase 15-A replay remains PASS for TP=1 and TP=2, including loss and both rank-local gradient shards. Candidate TP=1 sanity and TP=2 correctness were not run because candidates failed compile/ABI validation first. This is intentional failure containment, not a correctness pass.

## 8. Candidate ABI

The frozen target marker is:

```text
// AKA_TARGET_CONTRACT: vocab_parallel_cross_entropy:112959ca63020e9b
```

Required stream-aware entry points were:

```text
local_max_fp16_stream(..., cudaStream_t)
local_prepare_fp16_stream(..., cudaStream_t)
```

They launch only local kernels and perform no collectives. The caller owns real Megatron/NCCL orchestration.

## 9. Runtime/profile evidence

The Phase 15-A baseline NSYS commands and artifacts remain authoritative:

```text
<REMOTE_HOME>/tools/nsys-2026.2.1/target-linux-x64/nsys profile --trace=cuda,nvtx,osrt --sample=none -o ~/vocab_ce_tp1 <REMOTE_HOME>/venvs/lerobot-act/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone /tmp/vocab_ce_replay.py --bench

<REMOTE_HOME>/venvs/lerobot-act/bin/python -m torch.distributed.run --no-python --nproc_per_node=2 --standalone /tmp/profile_vocab_rank.sh
```

Baseline TP=2 traces contain local reduction/index/elementwise kernels and NCCL all-reduce kernels. Since no candidate passed ABI and correctness, candidate NSYS/NCCL traces do not exist and no candidate diagnosis is claimed.

## 10. Failure containment and negative tests

The remote evaluator compiles before launching `torch.distributed.run`; compile failure therefore cannot enter NCCL. ABI failure occurred after both ranks initialized but before candidate kernels or collectives, and torchrun propagated the rank failure and terminated the peer. The CLI rejects TP values other than 1 or 2. The replay harness rejects non-divisible vocab sizes. No rank was left hanging.

Invalid target IDs remain a caller precondition in the exact Megatron function; this phase does not add a production wrapper that validates arbitrary user targets. That limitation is recorded rather than hidden.

## 11. Incumbent and replay

`campaigns/targets/megatron_5be9626/vocab_parallel_cross_entropy/incumbent_manifest.json` records `NO_INCUMBENT`. Promotion requires a real TP=2 end-to-end score, and no episode reached it. Consequently there is no winning candidate replay and no candidate hash to promote.

## 12. Knowledge isolation

Only `knowledge/targets/megatron_5be9626/vocab_parallel_cross_entropy/` was used for this target. No standalone operator knowledge or SwiGLU target namespace was modified. The target summary records replay/baseline observations and the absence of optimization evidence; no speedup is presented.

## 13. Upstream integrity and prior evidence

The authoritative checkout was rechecked at HEAD `5be9626709af2722333bf54797c954c09edeada3` with a clean working tree. Phase 15-A replay artifacts, Phase 14-B/C/E SwiGLU evidence, and the Phase 14-G blocker remain auditable and were not mixed with this campaign.

## 14. Remaining limitations and next safe step

This phase proved the distributed campaign plumbing up to candidate ABI/build evaluation, but not a valid candidate. The immediate next step is a new, explicitly approved campaign turn that fixes the Agent prompt/schema contract so generated code exports the required portable, stream-aware ABI. Any such candidate must then pass TP=1, real TP=2 correctness on both ranks, and end-to-end TP=2 benchmarking before profiling or promotion. No backward optimization or collective-count change should be started until that gate passes.
