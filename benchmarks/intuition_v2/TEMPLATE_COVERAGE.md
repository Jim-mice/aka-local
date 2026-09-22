# Deterministic template coverage audit

Audited files: `lab/runtime/reasoning/hypothesis_planner.py`, `performance_model.py`, and `mechanism_memory.py`.

The current deterministic generator has direct branches for:

1. repeated global-memory reads: suggest extending value lifetime and avoiding a repeated read;
2. producer-consumer boundaries or read/write materialization overlap: suggest compatible fusion or intermediate removal;
3. missing kernel-launch/profile fields: emit a measurement hypothesis for launch, boundary, and traffic evidence;
4. operator fraction below 0.05: emit a system-ceiling/deprioritization measurement hypothesis.

The base retrieval ranker additionally scores existing records using repeated-read, materialization, launch count, synchronization, reuse, estimated bytes, working-set feasibility, support/contradiction evidence, and prior evidence status. These are ranking features, not novel mechanism generation.

The generator does not have explicit branches for algorithmic reformulation, recompute-versus-store decisions, ownership/remapping, decomposition changes, occupancy-aware rejection of a familiar transformation, or multi-scope opportunity comparison beyond the low-fraction reminder.

Held-out coverage intent:

| Case | Main challenge | Direct old-template hit? |
|---|---|---|
| A | algorithmic reformulation under numerical constraints | no |
| B | producer-consumer trap with measured occupancy/spill facts | surface trigger only; correct answer requires rejecting/down-ranking fusion |
| C | repeated-use pattern contradicted by register/spill evidence | surface trigger only; correct answer requires rejecting lifetime extension |
| D | competing opportunities across scopes and nontrivial system value | low-fraction branch can notice one fact, but cannot solve the comparison |
| E | missing bottleneck evidence with a misleading magnitude fact | measurement branch may trigger, but bait-vs-causal evidence discrimination is held out |

At least four of five cases therefore require behavior beyond selecting the existing positive templates. No v2 answer is added to Mechanism Memory or the planner.
