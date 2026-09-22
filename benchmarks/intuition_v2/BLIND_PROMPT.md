# Held-out blind planning prompt

Read only the selected case directory's `case.json`, `facts.json`, `contract.json`, and this prompt. Return JSON only using `planning_result.schema.json`.

Propose at most three evidence-grounded hypotheses or measurement hypotheses. You may reject, down-rank, or defer an attractive transformation when supplied facts contradict its preconditions. Preserve unknowns as unknowns. Do not write code, claim new measurements, add a mechanism record, or make a promotion decision.

Each hypothesis must include: `observation`, `mechanism`, `transformation`, `preconditions`, `expected_effect`, `risks`, `unknowns`, `required_evidence`, and `evidence_refs`. Every evidence reference must be an ID in the selected `facts.json`. If the facts are insufficient, say what measurement would distinguish the alternatives.

Do not read `evaluator_only`, previous results, deterministic planner output, or any other case. Do not assume the case has a known optimization answer.
