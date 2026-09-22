# Run from a fresh Codex session

1. Open a genuinely new Codex session. Do not start `codex.exe` from inside another Codex session, do not use nested app-server calls, and do not change model or user configuration.
2. Work read-only. Set the D repository as the working directory.
3. Choose exactly one case directory: `rmsnorm`, `gdn`, or `swiglu`.
4. Read that case's `case.json`, `facts.json`, `contract.json`, this directory's `BLIND_PROMPT.md`, and `planning_result.schema.json`. Do not read `evaluator_only` or any previous result slot.
5. Produce JSON only, save it as the selected result slot's `result.json`, and preserve the unmodified response as `raw_response.txt`.
6. Use one fresh context per run. Do not show previous outputs to later runs. Use three independent slots per case.

The external session must not add facts, answer keys, mechanism labels, benchmark results, code, or promotion decisions. If evidence is unknown, report it as unknown and request a measurement. After collection, run `scripts/collect_intuition_result.py`; only then run `scripts/evaluate_intuition_result.py`.
