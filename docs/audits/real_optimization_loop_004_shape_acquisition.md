# Real Optimization Loop 004 — Representative Shape Acquisition

## Timing evidence reconciliation

The authoritative raw artifact is:

`artifacts/integration/swiglu/real_loop_003/batched_swiglu_timing.json`

It records the final compliant protocol: `K=512`, 20 warmup batches, 50 measured batches, median measured window `27.518064498901367 ms`, and median per-invocation `53.74621972441673 µs`.

The `55.63945323228836 µs` value came from an earlier `K=8192` diagnostic run. Its raw batch file was later overwritten by the final K=512 run; the value remains only in the old derived `measurement_facts_003.json` and prior report final block. It is not authoritative. Derived artifacts and the Loop 003 report were corrected without rerunning the benchmark.

Reconciliation record: `artifacts/integration/swiglu/real_loop_004/timing_evidence_reconciliation.json`.

## Minimum operator replay contract

The isolated SwiGLU replay requires one authoritative configuration identity and these fields:

- hidden size;
- FFN hidden size;
- sequence length;
- micro-batch size;
- dtype;
- tensor-parallel size;
- sequence-parallel setting;
- gated activation type;
- bias setting;
- layout.

Checkpoint, tokenizer, dataset, and optimizer state are irrelevant to isolated tensor-shape replay once these fields are provided. They remain relevant to full training and nine-grid E2E.

Contract: `artifacts/integration/swiglu/real_loop_004/representative_shape_contract.json`.

## Local source audit

The pinned Megatron source and D repo contain real operator contracts and historical fixtures, but no authoritative target nine-grid configuration. The existing `nine_grid_readiness.json` explicitly records that the target configuration, checkpoint, tokenizer/data, topology, and exact invocation are missing.

The most concrete SwiGLU fixture is:

`targets/megatron_5be9626/swiglu/fixtures/fixture_metadata.json`

It records a historical traceable fixture with `S=128`, `B=2`, `H=1024`, FP16, TP=1, contiguous layout, gated SiLU, and `add_bias_linear=false`. It also identifies a V100 fixture. It does not establish that this is the requested nine-grid configuration, and it does not provide FFN hidden size or sequence-parallel setting under the same target identity.

The SwiGLU contract and replay contract provide formulas and callsites, not the missing target configuration values. The existing V100-labelled shape/contract material is read-only historical evidence and is not promoted to a nine-grid claim.

## Configuration identity

`nine_grid_shape_provenance.json` records `configuration_identity = null`. The available fixture identity is retained separately as an observed historical fixture identity, with ambiguous status for fields that are not proven to belong to the requested target.

No values from unrelated configurations were combined. No model dimensions were inferred from model names, papers, common 7B/14B/32B conventions, or Megatron examples.

## Nine-grid shape provenance

Core target fields are not complete under one authoritative identity:

- hidden size: observed as 1024 only in the historical fixture, not target-confirmed;
- FFN hidden size: missing for the target identity;
- sequence length/micro-batch/dtype/TP: observed in the historical fixture, not target-confirmed;
- sequence parallel: missing for the target identity;
- activation/bias/layout: available for the historical fixture, not enough to establish target identity.

Therefore the target representative replay is not ready.

## SwiGLU replay readiness

`REPRESENTATIVE_SHAPE_STATUS = PARTIAL`.

There is enough material to describe a traceable historical operator fixture, but not enough to construct a replay that can honestly be called representative of the requested nine-grid configuration. Consequently, `representative_swiglu_replay_spec.json` was not created.

Minimal external requirements are in `artifacts/integration/swiglu/real_loop_004/minimal_external_shape_requirements.json`. The short list is the target configuration identity plus hidden size, FFN hidden size, sequence length, micro-batch size, dtype, TP, and SP. Checkpoint/tokenizer/dataset/optimizer state are not requested for this isolated replay gate.

## Five-operator representative-shape matrix

`representative_operator_shape_matrix.json` records:

| Operator | Status | Reason |
|---|---|---|
| Dense Fused Attention | NOT_AVAILABLE | Protocol template has null model config/topology/shape values. |
| Vocab-parallel Cross Entropy | PARTIAL | Traceable TP=2 shape set exists, but no target nine-grid identity. |
| SwiGLU | PARTIAL | Historical fixture exists; target identity, FFN, and SP are incomplete. |
| Residual Add RMSNorm | PARTIAL | Multiple traceable shape suites exist, but no single target identity. |
| MoE Grouped GEMM | PARTIAL | MoE artifacts exist, but no same-identity target expert/router shape contract. |

No other fixed operator has a complete, same-identity representative shape either.

## Minimal external requirements

The current blocker is not checkpoint or data. It is authoritative configuration identity and the seven core shape/parallel fields. Supplying those fields from one traceable target configuration is sufficient to construct an operator replay spec; no full nine-grid E2E assets are needed for that isolated step.

## Project branch decision

`NEXT_PROJECT_BRANCH = WAIT_FOR_AUTHORITATIVE_SHAPE`.

The next action is to obtain the target configuration identity and required shape/parallel fields. Do not run a replay from the historical V100 fixture, do not guess missing FFN/SP values, and do not create an optimization candidate.

Loop 003 remains scoped to its frozen tiny local workload. This audit does not generalize its rejection to all SwiGLU implementations or all future nine-grid shapes.

## Final status

```text
TIMING_EVIDENCE_RECONCILED = PASS
AUTHORITATIVE_BATCHED_SWIGLU_US = 53.74621972441673
REPRESENTATIVE_SHAPE_STATUS = PARTIAL
CONFIGURATION_IDENTITY = null
HIDDEN_SIZE = null
FFN_HIDDEN_SIZE = null
SEQUENCE_LENGTH = null
MICRO_BATCH_SIZE = null
DTYPE = null
TP_SIZE = null
SP_ENABLED = null
CHECKPOINT_REQUIRED_FOR_REPLAY = NO
TOKENIZER_REQUIRED_FOR_REPLAY = NO
DATASET_REQUIRED_FOR_REPLAY = NO
REPRESENTATIVE_REPLAY_SPEC = NOT_CREATED
OTHER_OPERATOR_WITH_REPRESENTATIVE_SHAPE = null
NEXT_PROJECT_BRANCH = WAIT_FOR_AUTHORITATIVE_SHAPE
OPTIMIZATION_CANDIDATE_CREATED = NO
MECHANISM_MEMORY_MODIFIED = NO
NINE_GRID_E2E_EXECUTED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
```
