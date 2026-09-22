# Night Shift Report

Status: LOCAL SCOPE EXHAUSTED; EXTERNAL INTEGRATION REQUIRED

## Executive summary

The fixed five-operator program now has source-grounded Megatron integration
contracts, three independent OJ levels, end-to-end-driven promotion, a
structured Performance Reasoner, mechanism memory, evidence-aware hypothesis
ranking, an Agent-context boundary, and complete local blueprints for SwiGLU
and Dense Attention. No E2E numbers were invented and no GPU or remote system
was contacted.

## Time / phase accounting

- 00:22-00:30 +08:00: recovery state, source/import/commit checks, baseline
  reports, and initial regression diagnosis.
- 00:30-00:42: five contracts, L0/L1/L2, promotion policy, reasoner, memory,
  planner, context, and blueprints.
- 00:42-01:15: stretch work, initial policy hardening, collective-trace checks,
  regression recovery, architecture review, syntax/JSON checks, and docs.
- 01:32-02:00: resumed from durable state; hardened fail-closed numeric
  evidence, auditable hypothesis ranking, import isolation, self-contained
  restricted-filesystem tests, exact Megatron source contracts, and Agent
  context importability.
- 03:33-04:55: resumed from durable state; added source and bundle provenance,
  strict result-chain schemas, protocol/run/artifact binding, authority and
  profile-evidence scoping, strict JSON, external handoff templates, multi-rank
  mock traces, read-only verification CLIs, and repeated full regressions.

The planned window was not consumed artificially: all safe local mandatory and
stretch work was implemented and reviewed. Remaining work needs external
model/data/topology or a less restrictive test filesystem.

## Mandatory completion matrix

| Phase | Status | Evidence |
|---|---|---|
| Phase 0 baseline | PASS | P0 original 11/11 and P1 original 9/9 through the self-contained ACL-safe offline launcher |
| Phase 1 five-operator map | PASS | five JSON contracts plus design map |
| Phase 2 three-level OJ | PASS | L0/L1/L2 result models and mockable judges |
| Phase 3 E2E promotion | PASS | required E2E metrics, configurable policy, independent states |
| Phase 4 Performance Reasoner | PASS | `PerformanceFacts` and `Opportunity` |
| Phase 5 Mechanism Memory | PASS | sink extension, query, Episode 28 record |
| Phase 6 hypothesis ranking | PASS | evidence/risk/dataflow ranking and counterfactuals |
| Phase 7 Agent integration | PASS | hashed optional reasoner context on formal Agent turns |
| Phase 8 SwiGLU blueprint | PASS | static blueprint, mock harness, artifact/promotion plan |

## Five-operator integration map

Contracts cover Dense Fused Attention, Vocab-parallel Cross Entropy, SwiGLU,
Residual Add RMSNorm, and MoE Grouped GEMM at Megatron-LM commit
`5be9626709af2722333bf54797c954c09edeada3`. Each freezes source symbols,
forward graph, backward relevance, dtype/layout/shapes, distributed semantics,
replacement exclusions, correctness gates, fallback detection, and L0/L1/L2
metrics. A reusable validator rejects schema drift, unknown top-level fields,
empty required evidence, unpinned commits, and unsafe source paths. The
Megatron checkout was read-only and matched the expected commit. A source-audit
artifact records SHA-256 values for all 15 referenced source files and line
anchors for all five operators.
The source audit distinguishes unfused CE's MAX/SUM/SUM trace from native
fused CE's MAX/packed-SUM trace, and records that
`TEFusedResidualRMSNorm` alone does not perform the preceding residual add.

## Three-level OJ

- L0 returns `OPERATOR_PASS/FAIL` only after compile, contract, shape,
  correctness, stability, positive finite latency, a valid sample vector, and
  valid contract/candidate SHA-256 values.
- L1 returns `INTEGRATION_PASS/FAIL` only after the replacement marker,
  zero-fallback, full forward/backward, shapes, distributed invariants,
  required metrics, structured runner provenance, and any declared collective
  order pass.
- L2 returns `SYSTEM_PROVISIONAL`, `SYSTEM_QUALIFIED_ACCEPT`, or
  `SYSTEM_REJECT` from raw system metrics and structured qualification
provenance (protocol hash, distinct repeated run IDs, stability,
comparability, and artifact references). Bundle validation additionally binds
those references to the same SHA-256 manifest.

## End-to-end promotion policy

`KernelPromotion`, `IntegrationPromotion`, and `SystemPromotion` are stored
separately. Acceptance cannot skip a level. Required default E2E evidence
includes baseline/candidate iteration time, samples/s, tokens/s, peak memory,
convergence-proxy delta, loss delta, and gradient sanity. Missing metrics remain provisional; malformed
or non-finite metrics reject. Thresholds are configurable and raw inputs are
preserved. Verdict-to-promotion translation has one shared implementation, and
result payloads use strict JSON; the infinite Amdahl case is represented by an
explicit unbounded flag rather than `Infinity`. Bundle manifest and artifact
decoders also reject non-standard constants and duplicate object keys. An
explicit memory delta cannot override the value derived from baseline and
candidate peak-memory evidence.

## Performance Reasoner

`PerformanceFacts` represents dataflow, lifetime/residency, reads/writes,
bytes/FLOPs, reductions, synchronization, launches, fusion boundaries, reuse,
fixed/dynamic dimensions, mapping, profile evidence, unknowns, and E2E share.
Empty shapes and malformed lifetime/mapping entries are rejected. Magnitude is
intentionally categorical, never a fabricated percentage.

## Mechanism Memory

`MechanismStore` extends the P1 structured sink rather than creating another
database. The Episode 28 lifetime/HBM lesson is stored with v23/v28 observations
but marked `CAUSAL_UNPROVEN` because vectorization, specialization, and
unrolling were simultaneous confounders. Nested evidence and JSONL records are
strict-JSON validated.

## Hypothesis Planner

The planner ranks applicable mechanisms using repeated reads, unnecessary
materialization, estimated bytes saved, launch/synchronization opportunity,
reuse, working-set feasibility, profile support or contradiction, evidence
status, and risk. Its score factors are retained for audit.
It answers all requested counterfactuals, including the Amdahl ceiling from an
optional E2E operator fraction.

## Agent integration

Formal Agent turns can receive a canonical-JSON re-hashed
`performance_reasoning` payload with
facts, top opportunities, mechanisms, magnitude, risks, and evidence status.
The payload states that planning is advisory, implementation belongs to the
Agent, repairs belong to the bounded attempt loop, evaluation belongs to OJ,
and promotion remains controller-owned. The attach boundary rejects altered
authority maps, modified implementation instructions, unknown fields,
unbound opportunity records, and non-standard JSON numerics before hashing. SDK import is delayed until
`CodexAgentSession.start()`, so this boundary is offline-testable even when the
real Agent dependency is unavailable.

## SwiGLU blueprint

The initial boundary is `bias_swiglu_impl` for the declared non-TE,
non-weighted dispatch between TP FC1 and FC2. The blueprint defines
runtime shape capture, forward equation checks, activation gradients, full-MLP
L1 checks, invocation/fallback markers, L2 training metrics, artifacts, and
promotion states. Its static artifact manifest validator checks pinned commit,
declared paths, SHA-256 values, replacement marker, zero fallback, and all
promotion axes. The on-disk bundle validator checks path containment and real
digests, parses each JudgeResult, enforces L0-to-L2 prerequisite order, binds
verdicts to promotion states, requires qualified L2 evidence to reference
the hashed raw-metrics artifact, and matches its run IDs to that artifact's
baseline/candidate records. It also rejects incomplete core check sets and
verdict/check/reason inconsistencies, and binds the protocol hash to the hashed
`provenance.json`. A real training command is intentionally unresolved until
the nine-grid model, data, topology, and measurement policy are provided.
Empty contract/shape/provenance placeholders are rejected: these artifacts
must carry target identity, captured dimensions/strides/config, and structured
runner/environment/protocol metadata.
The checked-in E2E protocol template deliberately leaves the real training
command, model/data references, topology, run IDs, and qualification evidence
unset; null fields cannot qualify.

## Stretch work

- Dense Attention blueprint: complete for the declared dense/no-mask/p=0 path.
  Its external handoff template is policy-aligned and intentionally unfilled.
- Amdahl ceiling: implemented and tested.
- Ablation planner: generates baseline, isolated main effects, leave-one-out
  marginal effects, and combined reproduction.
- Architecture review: complete; low-risk fixes applied and risky legacy
  migration deferred.
- Read-only verification CLIs: source audit returns `SOURCE_AUDIT_PASS`; the
  checked-in SwiGLU structural fixture returns `BUNDLE_PASS`.
- Because the D snapshot has no Git metadata, 35 stable core artifacts have a
  separate SHA-256 inventory; its read-only verifier returns
  `NIGHT_ARTIFACT_PASS`. Mutable STATE/JOURNAL/report files are intentionally
  excluded.

## Files changed

Primary additions are under `targets/megatron_5be9626/integration_contracts/`,
`lab/runtime/evaluators/`, `lab/runtime/reasoning/`,
`lab/runtime/blueprints/`, `config/protocols/`, `scripts/`, `docs/design/`, and
`docs/night_shift/2026-09-21/`.
`lab/runtime/agent/codex_session.py` gained the optional reasoner-context
boundary. No historical candidate source was changed.

## Tests

- Night Shift architecture suite after final local hardening: 21/21 PASS.
- P0 original offline suite: 11/11 PASS.
- P1 original offline suite: 9/9 PASS.
- New Python sources and CLIs: syntax-compiled successfully.
- Contract/state/policy/fixture JSON and JOURNAL JSONL: parsed successfully.
- Pinned Megatron source recheck: `SOURCE_AUDIT_PASS`.
- Checked-in structural operator bundle: `BUNDLE_PASS`.
- Stable Night Shift artifact inventory: `NIGHT_ARTIFACT_PASS` (35 files).

## Blockers

- The D snapshot has no `.git`, so Git provenance/diff is unavailable.
- Failed early native tempfile runs left sandbox-owned `tmp*` directories that
  this session cannot read or safely remove. The self-contained launcher now
  uses inherited-ACL fixtures and completes all original P0/P1 tests.
- `openai_codex` is not installed in the local venv, so the real Agent session
  was not instantiated; its deterministic context builder is tested offline.
- The downloaded snapshot still lacks the older `lab.core` package noted by P1.

## External integration required

Real L1/L2 qualification needs the nine-grid model/config, dataset, training
command, distributed topology, installed Megatron dependencies, and approved
GPU access. No attempt was made to obtain or contact them.

## Remaining risks

- Legacy standalone paths still emit `PROMOTE` / `QUALIFIED_ACCEPT`. They must
  use the new explicit kernel-only adapter during a future reviewed migration;
  it is deliberately not auto-wired and cannot set Integration/System.
- CE and MoE need real multi-rank collective traces; only deterministic mock
  trace ordering across complete mock rank sets was exercised locally.
- Scientific owners must set task-specific loss and memory thresholds.

## Recommended next daytime actions

1. Supply a real nine-grid Megatron command and capture source-point shapes.
2. Run SwiGLU L1 first, then controlled L2 baseline/candidate training windows.
3. Migrate legacy standalone decisions through an explicit Kernel-only adapter.

## Final safety check

```text
D_REPO_ONLY = PASS
C_AKA_MOTHER_MODIFIED = NO
MEGATRON_MODIFIED = NO
REMOTE_V100_USED = NO
BIV150_USED = NO
CUDA_BENCHMARK_USED = NO
HISTORICAL_CANDIDATES_MODIFIED = NO
P0_REGRESSION = PASS (11/11 original tests)
P1_REGRESSION = PASS (9 original tests)
SOURCE_AUDIT = PASS
OPERATOR_BUNDLE_AUDIT = PASS
NIGHT_ARTIFACT_SHA256 = PASS (35 stable files)
```
