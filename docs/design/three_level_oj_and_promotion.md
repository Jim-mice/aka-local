# Three-level OJ and promotion

The judge hierarchy separates three questions that standalone benchmarks
cannot collapse:

1. L0 Operator OJ checks compile, contract, declared shapes, standalone
   correctness, positive finite latency, a non-empty finite sample vector, and
   stability. Passing evidence also requires canonical 64-hex contract and
   candidate hashes. It returns
   `OPERATOR_PASS` or `OPERATOR_FAIL`.
2. L1 Integration OJ proves the target path was actually replaced, checks the
   expected marker and zero silent fallback, full forward/backward behavior,
   real shapes, finite nonnegative required memory/timing metrics, and exact collective traces
   where declared. A numeric zero fallback count and structured runner ID plus
   artifact references are mandatory rather than inferred from a boolean. Its
   runner provenance must match the controller-declared contract and candidate
   SHA-256 identities.
   Optional multi-rank trace evidence requires the exact declared rank set and
   identical expected collective order on every rank. It returns
   `INTEGRATION_PASS` or `INTEGRATION_FAIL`.
3. L2 End-to-End OJ preserves raw baseline/candidate metrics and derives
   iteration speedup, throughput gain, memory delta, and optionally the Amdahl
   ceiling. It returns `SYSTEM_PROVISIONAL`, `SYSTEM_QUALIFIED_ACCEPT`, or
`SYSTEM_REJECT` under a versioned policy.

`SYSTEM_QUALIFIED_ACCEPT` cannot be requested with a bare boolean. The L2
judge requires structured `QualificationEvidence`: a 64-hex protocol hash,
distinct baseline and candidate run IDs meeting the configured repeat count,
stable/comparable flags, and artifact references. Missing qualification stays
provisional; malformed qualification rejects.

Blueprint result bundles add a second, offline integrity boundary. Every
declared artifact is read from a path confined to the bundle root and checked
against its SHA-256. L0/L1/L2 result JSON must match the shared `JudgeResult`
shape, valid level/verdict pairs, the complete core checks for its level,
verdict/check/reason consistency, prerequisite order, and the three promotion
axes. L2 must preserve the policy's complete required-metric check set. A
qualified L2 result must contain valid repeated-run
qualification provenance, and every `artifact_ref` must name an artifact in
the same hashed manifest. Its protocol hash must equal the manifest digest for
`provenance.json`, and baseline/candidate IDs must match `l2/raw_metrics.json`.
The L0 contract hash must equal the actual hashed `contract.json`; its candidate
hash must equal the manifest identity, and both identities must also appear in
L1 runner provenance and the L2-bound measurement provenance artifact.
This validates evidence linkage, not the scientific
truth of externally produced measurements.

`PromotionState` stores `KernelPromotion`, `IntegrationPromotion`, and
`SystemPromotion` independently. It enforces prerequisite ordering, never
copies an accepted lower-level state upward, and invalidates downstream states
if a prerequisite is later rejected or made provisional. Any evaluated L1
state requires accepted L0, and any evaluated L2 state requires accepted L1;
this applies to rejected/provisional states as well as accepted states. Thus L0
acceptance is useful inner-loop evidence and is not a project reward.
`PromotionState.apply_result()` and bundle validation share one canonical
verdict-to-axis mapping, preventing a second status translation path.

The initial E2E policy is intentionally neutral and configurable. It requires
gradient sanity and repeated qualification while leaving scientific
loss/memory thresholds unset until an owner declares them. Missing evidence
produces a provisional result; malformed, non-finite, non-positive rate/timing,
non-boolean gradient, or non-explicit qualification evidence rejects rather
than passing.
Raw metric dictionaries must be strict-JSON serializable. The `fraction=1`
Amdahl case is stored as `max_possible_e2e_speedup: null` plus an explicit
`max_possible_e2e_speedup_unbounded: true`, never non-standard JSON `Infinity`.
When an input also supplies `memory_delta_bytes`, it must be finite and equal
the delta derived from baseline/candidate peak memory; it cannot override the
source metrics or bypass a configured memory threshold.
