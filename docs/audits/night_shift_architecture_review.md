# Night Shift Architecture Review

## Scope

This is a static, offline review of P0/P1 and the Night Shift additions. No
remote evaluator, CUDA benchmark, historical candidate, mother repository, or
Megatron source was modified.

## Findings

1. The new `JudgeResult` is intentionally shared by L0/L1/L2, avoiding three
   nearly identical result schemas. `PromotionState` is the single new source
   of truth for Kernel, Integration, and System promotion axes.
2. Mechanism memory extends `StructuredKnowledgeSink`; it does not introduce a
   second persistence engine. The older strategy-tag readers remain a legacy
   compatibility path and are not treated as mechanism evidence.
3. `augment_authoritative_context` re-hashes the snapshot after attaching
   performance reasoning. The Agent receives the data but retains no evaluator
   or promotion authority.
4. The old standalone `controller_policy.decide()` and V100 campaign still use
   `PROMOTE` / `QUALIFIED_ACCEPT`. Reusing those values as `SystemPromotion`
   would be an unsafe authority leak. They remain explicitly legacy L0-like
   decisions until a future integration migration maps them through L1/L2.
5. The downloaded D snapshot lacks `.git` and the older `lab.core` package.
   Reconstructing that subsystem tonight would be a broad, unrelated refactor.
6. `LongHorizonRunner._context` contains unreachable compatibility code after
   an unconditional return. Removing it is low value while `lab.core` is
   absent, so it is recorded rather than edited.

## Low-risk fixes made

- Added explicit no-fallback markers and tests for both blueprints.
- Added policy loading plus samples/s derivation while preserving all raw E2E
  metrics.
- Kept absent measurements as `None` and provisional rather than passing.
- Reject non-finite/non-positive performance inputs, non-boolean gradient
  evidence, out-of-range Amdahl fractions, and non-explicit qualification.
- Expose auditable ranking factors for estimated bytes saved, launch and sync
  opportunity, reuse, working-set feasibility, profile support/contradiction,
  prior evidence, and risk.
- Made the restricted-filesystem regression launcher self-contained and
  tightened import isolation from the repository root to the actual `lab/`
  package root; the original P0 11 and P1 9 tests now pass unchanged.
- Replaced boolean-only L2 qualification with structured protocol/run/artifact
  provenance and added fail-closed blueprint artifact-manifest validation.
- Bound every blueprint artifact to its on-disk SHA-256, constrained artifact
  paths to the bundle root, checked result verdicts against promotion state and
  prerequisite order, and required L2 qualification references to resolve to a
  hashed raw-metrics artifact.
- Bound qualification run IDs to the baseline/candidate records in that raw
  artifact, rejected verdict/check/reason inconsistencies, required prerequisite
  acceptance for every evaluated downstream promotion state, and added a
  reusable fail-closed validator for the five pinned integration contracts.
- Made `PerformanceFacts` and ablation inputs reject malformed numeric and
  structural values before ranking or experiment-plan generation.
- Scoped mechanism-specific byte/working-set profile evidence, rejected
  altered Agent authority maps and non-standard JSON numerics, bound the L2
  protocol hash to `provenance.json`, and recorded pinned source hashes/anchors
  for every file used by the five contracts.
- Centralized verdict-to-promotion translation, made L2 raw/derived results
  strict-JSON safe including the unbounded Amdahl case, and rejected empty
  contract, shape-manifest, or measurement-provenance bundle artifacts.
- Added an explicit, unconnected legacy standalone adapter that can populate
  only `KernelPromotion`; it rejects System verdicts and cannot infer L1/L2.
- Added deterministic per-rank collective-trace checks for CE and MoE mocks;
  these validate rank/order schemas but remain distinct from real collectives.
- Made both manifest and artifact JSON decoding fail closed on non-standard
  constants, duplicate object keys, invalid UTF-8, and non-object manifests.
- Made E2E memory delta derived evidence: a caller-supplied delta must be
  finite and match baseline/candidate peak memory instead of overriding it.
- Tightened PerformanceFacts and Mechanism Memory container/evidence types,
  and made mechanism JSONL reject ambiguous or non-standard JSON.
- Froze the Agent implementation instruction, rejected unknown context fields,
  and required each ranked opportunity to bind to a supplied mechanism record.
- Bound the actual contract artifact digest and one candidate SHA-256 across
  L0, L1 runner provenance, the bundle manifest, and L2 measurement provenance.

## Deferred migration TODOs

- Wire the explicit legacy adapter at a reviewed migration point; it currently
  remains deliberately unconnected and never infers Integration/System.
- Wire real captured Megatron shape manifests into the static blueprints.
- Remove the unreachable legacy context block only after the complete
  `lab.core` package is restored and its tests run.
