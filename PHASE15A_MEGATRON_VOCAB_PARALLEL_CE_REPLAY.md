# Phase 15-A — Megatron Vocab-Parallel Cross Entropy Replay

## Verdict

PASS for the replay objective. The exact Megatron implementation was executed at commit `5be9626709af2722333bf54797c954c09edeada3`. TP=1 and real two-process TP=2 replay both matched a full-vocabulary PyTorch oracle for loss and local gradient shards. Two V100 GPUs and NCCL were verified. No CUDA candidate or Agent campaign was started.

## 1. Exact source and call chain

The source-grounded path is:

```text
Megatron language-module loss path
  -> megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy
  -> _VocabParallelCrossEntropy.forward
  -> _VocabParallelCrossEntropy.backward
```

The implementation is in `<LOCAL_USER_HOME>\projects\megatron-lm\megatron\core\tensor_parallel\cross_entropy.py`, with fused cross-entropy call-site support in `megatron/core/fusions/fused_cross_entropy.py` and the language-module integration in `megatron/core/models/common/language_module/language_module.py`.

## 2. Exact semantics

For `N=S*B`, global vocabulary `V`, and TP size `T`, each rank receives contiguous local logits `[N,V/T]`. Targets are global token IDs `[N]`; rank `r` owns `[rV/T,(r+1)V/T)`.

Forward source order:

```text
local logits cast/max in float
  -> MAX all-reduce(local max)
  -> subtract global max
  -> local target mask and masked target index
  -> local target-logit gather
  -> SUM all-reduce(predicted target logit)
  -> exp logits and local exp sum
  -> SUM all-reduce(exp denominator)
  -> log(denominator) - predicted logit
```

The current source therefore has three mathematically required forward all-reduces: MAX, SUM target logit, and SUM denominator. Backward uses saved exp/softmax, target mask, and masked target; it subtracts one only on the owning rank and has no collectives.

## 3. Source verification and safety

`<LOCAL_USER_HOME>\projects\megatron-lm` was verified at HEAD `5be9626709af2722333bf54797c954c09edeada3`; `git status --short` was empty. No authoritative Megatron file was modified. The aka-local Git-root anomaly was respected; no reset, clean, parent `.git` deletion, or nested repository initialization was used.

## 4. TP=1 and TP=2 runtime setup

The harness initializes NCCL with `torch.distributed.run`, sets the CUDA device from `LOCAL_RANK`, and initializes Megatron model-parallel state with tensor model parallel size equal to world size and pipeline size one. The remote environment is the exact isolated Megatron source copy `~/aka_targets/megatron-lm-5be9626/`, using Python `3.10` environment `lerobot-act`.

The two-GPU audit found two Tesla V100-PCIE-16GB devices. A minimal NCCL all-reduce smoke test passed on both ranks. The first smoke attempt exposed a harness device-selection bug; after setting `torch.cuda.set_device(LOCAL_RANK)`, both ranks reduced successfully. This was corrected in the harness, not in Megatron.

## 5. Shapes and dtype

Official replay configurations were derived from the mapped `[S,B,V/TP]` contract:

| S | B | global V | TP=2 local V | dtype |
|---:|---:|---:|---:|---|
| 8 | 1 | 32 | 16 | FP16 |
| 32 | 2 | 64 | 32 | FP16 |
| 128 | 2 | 128 | 64 | FP16 |

Logits and local gradients are FP16 fixtures. The source casts logits for max/reduction arithmetic to FP32; the loss is FP32. Targets are global `torch.long` IDs.

Deterministic fixture metadata is in `targets/megatron_5be9626/vocab_parallel_cross_entropy/fixture_metadata.json`.

## 6. Correctness evidence

The full-vocabulary oracle was computed with PyTorch cross entropy from the same deterministic global logits. For TP=2, the oracle was used only for validation; the replay execution supplied rank-local logits and used the real Megatron NCCL process group.

Both TP=1 and both TP=2 ranks passed all three configurations. Worst observed values across the runs were:

| Quantity | Worst observed |
|---|---:|
| loss max absolute error | 0.00327778 |
| loss max relative error | 0.0004330417 |
| local gradient max absolute error | 0.000488281 |
| local gradient max relative error | 0.00390625 |

The FP16 tolerances are `atol/rtol=0.002` for loss and `0.03` for gradients. The small loss absolute error is consistent with FP16 input/oracle comparison while source reductions use FP32.

Raw replay logs are under `targets/megatron_5be9626/vocab_parallel_cross_entropy/tp1.txt` and `tp2.txt`.

## 7. Timing boundary and baseline

The benchmark interval is a CUDA-event interval around the real Megatron forward plus backward operator. It includes local compute and mandatory TP collectives, and excludes process startup, imports, fixture generation, model-parallel initialization, and unrelated model work. Each sample performs a fresh detached local-logit leaf and backward; ranks synchronize with barriers for TP=2.

| configuration | TP=1 mean us | TP=1 std/CV | TP=2 mean us, rank 0 | TP=2 std/CV, rank 0 |
|---|---:|---:|---:|---:|
| [8,1,32] | 2002.34 | 65.40 / 0.0327 | 2774.02 | 41.42 / 0.0149 |
| [32,2,64] | 1980.62 | 38.81 / 0.0196 | 3138.56 | 388.01 / 0.1236 |
| [128,2,128] | 1948.26 | 45.59 / 0.0234 | 2812.93 | 20.33 / 0.0072 |

These are replay baselines, not optimization scores. The TP=2 rank-1 measurements are retained in `tp2_bench.txt` for audit.

## 8. REAL NSYS evidence

Commands used:

```text
<REMOTE_HOME>/tools/nsys-2026.2.1/target-linux-x64/nsys profile --trace=cuda,nvtx,osrt --sample=none -o ~/vocab_ce_tp1 <REMOTE_HOME>/venvs/lerobot-act/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone /tmp/vocab_ce_replay.py --bench

<REMOTE_HOME>/venvs/lerobot-act/bin/python -m torch.distributed.run --no-python --nproc_per_node=2 --standalone /tmp/profile_vocab_rank.sh
```

The TP=1 stats are in `tp1_nsys_stats.txt`; TP=2 rank stats are in `tp2_rank0_nsys_stats.txt` and `tp2_rank1_nsys_stats.txt`.

TP=1 runtime contains PyTorch reduction, index, exp/normalization, and backward kernels. TP=2 contains the same local kernel families plus `ncclDevKernel_AllReduce_Sum_f32_RING_LL`. This is direct runtime evidence of distributed communication. The source predicts three forward collectives; the NSYS aggregate shows the NCCL SUM family and CUDA API/launch activity. MAX may use the NCCL reduction kernel family in the trace depending on aggregation/name grouping and is not inferred beyond the captured evidence.

## 9. Compute/communication decomposition

The replay boundary decomposes into local max/reduction, MAX synchronization, local target extraction and exponentiation, SUM target-logit synchronization, local denominator reduction, SUM denominator synchronization, loss formation, and local backward gradient generation. The backward path is local-only at this commit.

For the three collective payloads, each rank contributes an `[N]` FP32 vector: 32, 128, and 256 elements respectively for the official configurations, or 128 B, 512 B, and 1024 B per collective. TP=2 NCCL ring all-reduce has two ranks and therefore unavoidable synchronization/communication at each of the three points. The exact source ordering prevents treating these as removable local reductions.

## 10. Runtime versus static mapping

Static source mapping and runtime agree: TP=1 is a multi-kernel local reduction/index/elementwise graph; TP=2 adds NCCL all-reduce kernels between those local stages. No fake local reduction or single-process emulation was used. The runtime is not one fused CE kernel.

## 11. Research-only future boundary

`targets/megatron_5be9626/vocab_parallel_cross_entropy/candidate_boundary.json` is explicitly marked `research_only_not_active`. Plausible future work is local kernel fusion and temporary reduction reduction around the mandatory collectives, or carefully verified overlap. It does not propose removing the three mathematically required forward all-reduces.

The replay contract is `targets/megatron_5be9626/vocab_parallel_cross_entropy/replay_contract.json`, hash `d8ea9db11871996c`. The target namespace is `knowledge/targets/megatron_5be9626/vocab_parallel_cross_entropy/`; no SwiGLU or standalone operator namespace was modified.

## 12. CLI and controlled failures

The target-oriented CLI now exposes:

```text
python -m lab.cli target --target vocab_parallel_cross_entropy --action inspect --tp 1
python -m lab.cli target --target vocab_parallel_cross_entropy --action replay --tp 1
python -m lab.cli target --target vocab_parallel_cross_entropy --action replay --tp 2
python -m lab.cli target --target vocab_parallel_cross_entropy --action benchmark --tp 2
python -m lab.cli target --target vocab_parallel_cross_entropy --action profile --tp 2
```

The CLI rejects TP values other than 1 or 2 before launch. The harness rejects non-divisible global vocabulary sizes before constructing a shard. Targets are generated in the valid global range. The exact Megatron function itself assumes valid target IDs and does not provide a general user-facing range-validation layer; therefore invalid target IDs remain a caller precondition and are documented as a limitation rather than falsely reported as a Megatron rejection. NCCL initialization failure is delegated to torchrun with a bounded SSH command timeout.

## 13. Prior-work and upstream integrity

Phase 14-B replay, Phase 14-C forward replay, Phase 14-E backward replay, and the Phase 14-G blocker report remain present and were not rewritten. No Agent campaign, candidate directory, incumbent, or score history was created for this target. The authoritative Megatron checkout remains at the required commit with a clean working tree.

## 14. Remaining blockers

This phase does not establish performance at production Megatron vocabulary sizes or sequence/batch configurations; it establishes the distributed boundary and a reproducible TP=1/TP=2 harness. Larger-shape NCCL scaling, exact production call-site integration under a full language model, and safe local-kernel/communication fusion remain future work. No optimization should begin until those measurements are expanded and the distributed contract remains unchanged.
