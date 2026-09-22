# Final held-out planning prompt

Read only this prompt, the selected case directory's `case.json`, `facts.json`, `contract.json`, and `planning_result.schema.json`. Return JSON only.

Propose at most three evidence-grounded mechanism or measurement hypotheses. A mechanism hypothesis must explain the observation, causal reason, concrete transformation, semantic preconditions, risks, unknowns, and required evidence. Do not write code, claim new measurements, add a knowledge record, or make a promotion decision. Preserve unknown fields as unknown. Evidence references must be IDs from `facts.json`.

Do not read any evaluator-only file, prior benchmark result, planner implementation, or other case. Do not assume that an obvious local kernel transformation is the correct answer; reason from the supplied mathematical contract and constraints.
