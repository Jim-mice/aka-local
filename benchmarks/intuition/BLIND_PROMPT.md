# Blind mechanism-planning prompt

You are given one case directory containing `facts.json`, `contract.json`, and no proposed solution. Read only those files.

Return one JSON object matching `planning_result.schema.json`. Propose at most three mechanism-level hypotheses grounded in the supplied facts. Do not write code, benchmark results, contract decisions, promotion decisions, or new measured facts. If a fact is unknown, preserve it as unknown and list what measurement would be required.

Every hypothesis must include: `observation`, `mechanism`, `transformation`, `preconditions`, `expected_effect`, `risks`, `unknowns`, `required_evidence`, and `evidence_refs`. Evidence references must use IDs present in `facts.json`; do not invent measurements.

The response must be JSON only. Do not infer or mention any answer key, prior run, planner output, or hidden evaluator metadata.
