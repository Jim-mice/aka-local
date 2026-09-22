# Performance Reasoner and Mechanism Memory

`PerformanceFacts` is the structured pre-implementation representation. It
records operator, shape, dtype, dataflow, tensor lifetimes and residency,
global reads/writes, byte/FLOP estimates, reductions, synchronization, launch
count, producer-consumer boundaries, reuse, fixed/dynamic dimensions, parallel
mapping, profile evidence, explicit unknowns, and optional E2E profile data.
Its constructor rejects empty or malformed dimension maps, non-string
lifetime/mapping entries, booleans/non-integers in byte/FLOP/launch estimates,
negative counts, and invalid structured containers before they can influence
ranking.

`HypothesisPlanner` answers the required counterfactual questions and ranks
stored mechanisms using applicable evidence: repeated reads, materialization,
estimated bytes saved, launch and synchronization removal, reuse, profile
support/contradiction, working-set feasibility, prior evidence, and risk. Each
score exposes these factors rather than only returning an opaque rank. Its
magnitude vocabulary is only `LARGE`, `MEDIUM`, `SMALL`, or `UNKNOWN`; it does
not manufacture percentages. The ordering is data-dependent rather than a
global hard-coded list. Profile estimates may be scoped by `mechanism_id`; when
that map is present, byte-saving and working-set evidence is never borrowed by
another candidate mechanism.

`MechanismStore` extends the existing `StructuredKnowledgeSink` JSONL path.
Records contain Pattern, Symptoms, Mechanism, Transformation, Preconditions,
ExpectedEffects, Risks, CounterEvidence, MeasuredEvidence, Applicability, and
Provenance. Record construction rejects empty identifiers/text, malformed
collections, and non-enum evidence status; JSONL query errors identify the
invalid record line instead of silently treating it as evidence. Nested
mechanism evidence must be strict-JSON serializable, and JSONL reads reject
non-standard constants and duplicate keys. Episode 28 is seeded as repeated immutable input consumption across
a reduction, redundant HBM traversal, and extended input lifetime. Its status
is deliberately `CAUSAL_UNPROVEN` because vectorization, specialization, and
unrolling changed simultaneously.

The Agent receives a hashed `performance_reasoning` context containing facts,
ranked opportunities, relevant mechanisms, magnitude, risks, and evidence
status. The authority declaration remains deterministic: the planner proposes,
the Agent implements, the bounded attempt loop repairs, the OJ evaluates, and
the controller owns promotion. Attachment revalidates the exact canonical
authority map, fixed implementation instruction, opportunity-to-mechanism
bindings, allowed field set, and strict-JSON serializability before re-hashing,
so a caller cannot delegate promotion or inject instructions, extra fields, or
`NaN`/`Infinity` through a custom payload.
