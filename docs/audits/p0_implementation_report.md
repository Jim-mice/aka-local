# P0 Implementation Report

## Files changed

Production changes are limited to the D-repository copy:

- `lab/__init__.py` — makes `lab` a regular package so Python cannot merge a foreign namespace-package contribution.
- `scripts/run_lab.py` — repository-root-derived launcher with fail-closed `lab` module and namespace-path checks.
- `lab.ps1` — calls the launcher and requires either the D-repo `.venv` or an explicitly supplied interpreter; it has no global editable-install fallback.
- `lab/runtime/evaluators/contract_validation.py` — canonical semantic contract, deterministic SHA-256, evaluation fingerprint, source/interface validation, and legacy classification.
- `lab/runtime/evaluators/qualification.py` — declared repeated-measurement qualification and bimodality checks.
- `lab/runtime/evaluators/phase8d.py` — writes version-2 manifests with distinct semantic and evaluation identities while retaining the old field as explicitly legacy.
- `lab/runtime/evaluators/remote_v100.py` — assigns provenance to each run and removes the duplicate primary-shape measured evaluation.
- `lab/runtime/evaluators/remote_v100_campaign.py` — validates the contract locally before evaluator use and applies the provisional/qualification decision gate.
- `operators/rms_norm_v100_cuda/metadata.json` — records semantic fields needed by the canonical contract; no candidate source changed.
- `config/environments/v100_sm70/evaluation.json` — declares qualification policy rather than embedding it in code.
- `lab/tests/test_p0_integrity.py` — offline regression coverage.

No file under `C:\Users\38154\projects\aka-local` was read as project source or modified.

## Import isolation

`scripts/run_lab.py` derives `PROJECT_ROOT` from its own path, puts that root at the front of `sys.path`, then verifies every already loaded `lab`/`lab.*` module.  A module `__file__`, and every namespace `__path__` entry when present, must resolve under the D-repository root.  Any foreign preloaded module fails with `REFUSING_TO_RUN` rather than falling back.

`lab.ps1` now invokes this launcher.  Its default interpreter is only `$ROOT\.venv\Scripts\python.exe`; if that interpreter is absent, the command fails with an instruction to pass `-Interpreter`.  It does not silently use a global Python environment.

## Canonical contract design

`CanonicalContract` serializes a normalized JSON object with sorted keys and compact separators, then computes `semantic_contract_sha256` using SHA-256.  The semantic payload includes operator identity/version, interface and entry symbol, ordered typed arguments and roles, dtype, semantic identifier/semantics, frozen shapes, tolerance, required marker, and target architecture.

`evaluation_fingerprint` is separate.  It covers scoring shapes and execution configuration such as warmup, iterations, compiler, and flags.  It must not be interpreted as a semantic contract.

For compatibility, `phase8d.build_episode_manifest` retains the old `contract_hash`, but labels it `legacy_contract_hash_kind: evaluation_shapes_only`.  New manifests add `semantic_contract_sha256`, `evaluation_fingerprint`, `interface_entry`, `interface_arguments`, and `result_schema_version: 2`.

## Contract validation

`validate_episode_contract` checks the required source marker, exported entry symbol, ordered argument signature, hypothesis operator/version/semantic hash/interface, manifest operator/semantic hash/interface, and canonical metadata.  A mismatch returns `REJECT_CONTRACT` with artifact, field, expected, and actual values.

`run_evaluation_for_episode` executes this local validation before calling `evaluate_v100`; therefore no SSH, SCP, `nvcc`, or remote job is reached after a mismatch.  The default direct remote-evaluator entry applies the same source/contract gate, so a CLI or legacy caller cannot bypass it.  Existing legacy manifests without `semantic_contract_sha256` are classified `LEGACY_UNVERIFIED_CONTRACT` and are left untouched: the evaluator returns before creating or rewriting result, decision, or manifest files.  This protects episode 28 and other historical evidence from in-place upgrade.

## Evaluator provenance fix

`evaluate_candidate_multi_shape` no longer performs a separate primary-shape "compile" call.  In the legacy remote helper, that call performed a full compile/correctness/benchmark job, so the old flow measured the primary shape twice and mixed evidence from job one with shape data from job two.

The revised flow has one measured remote job per requested shape (unless an explicitly requested profile creates its separate profile-only job).  Each `runs[]` entry records `run_id`, `job_id`, candidate SHA-256, shape, operator, timestamp, compile evidence, correctness evidence, and benchmark evidence.  The primary `evidence` field is copied from the exact primary run, so its `run_id` matches the primary shape result.

## Benchmark qualification design

An initial result above the incumbent is now only `PROVISIONAL`.  The declared qualification policy in `config/environments/v100_sm70/evaluation.json` requests five repeats, median scoring, a maximum CV, round-robin shape order, and an explicit bimodality policy.

`qualification.py` preserves all raw samples and calculates median, min/max, mean, sample standard deviation, CV, and the largest normalized gap in sorted samples.  It marks data bimodal when that gap meets the configured separation threshold and both sides have the configured minimum cluster size.  Stability requires both CV compliance and no bimodality.  A qualified median must still exceed the incumbent before the campaign writes `QUALIFIED_ACCEPT`; otherwise the candidate remains `PROVISIONAL_UNSTABLE`/`PROVISIONAL`.

## Result schema changes

New multi-shape evaluator output has `result_schema_version: 2`, a `runs` list, `semantic_contract_sha256`, `evaluation_fingerprint`, and a `qualification` object.  It distinguishes compile, correctness, benchmark, qualification, and promotion decisions without rewriting old result files.

## Backward compatibility

Old readers can retain use of `contract_hash`, `shapes`, and legacy aggregate fields.  New code never silently upgrades or trusts an old shape-only hash as semantic identity.  A legacy episode is readable for audit and explicitly unverified for new promotion/evaluation work.

## Tests added

`lab.tests.test_p0_integrity` is offline-only and covers:

- canonical hash determinism and semantic-field sensitivity;
- source/hypothesis mismatch rejection;
- a mock campaign proving a contract mismatch makes zero evaluator calls;
- a direct remote-evaluator call proving a malformed source is rejected before its SSH helper can be reached;
- immutable handling of a legacy episode manifest;
- four shapes producing exactly four measured evaluator instances, with a single primary invocation and shared provenance;
- foreign `PYTHONPATH` isolation and fail-closed preloaded foreign `lab` handling;
- a bimodal synthetic series (`9.0, 9.1, 22.0, 9.0, 22.1`) rejected despite its aggregate appearance;
- a stable synthetic series accepted and round-robin shape ordering.

## Test results

With `lab.__path__` verified as `D:\Users\38154\Downloads\aka-local-main\aka-local-main\lab`, the following completed successfully:

```text
python -m py_compile scripts/run_lab.py lab/__init__.py ... lab/tests/test_p0_integrity.py
python -m unittest lab.tests.test_p0_integrity -v

Ran 11 tests ... OK
```

**P0_REPORT_CORRECTION:** the report initially said `Ran 10 tests ... OK`.
The test module contains and executes 11 test methods; the mismatch was a
documentation count left behind after the final direct-entry contract-gate
test was added.  No P0 production behavior changed for this correction.

The tests use temporary local fixtures and fake evaluators only.  They do not open SSH, invoke `nvcc`, access V100, run CUDA, or alter any historical candidate.

## Remaining risks

- The remote `eval.sh` protocol was not changed or exercised.  The provenance behavior is validated with a fake evaluator; it needs a later controlled remote integration validation before using it for promotion.
- The current qualification policy is explicit and configurable, but its numerical thresholds are an initial policy choice that needs scientific-owner review.
- Profile collection can still be a separate remote profile job when requested; it is intentionally not counted as a benchmark repeat.
- The campaign's legacy knowledge/promotion consumers have compatibility fields, but the P1 knowledge redesign and same-session repair loop remain deferred.

## Deferred P1/P2 work

Not implemented in this P0 change:

- persistent same-Agent implementation repair and long-horizon remote orchestration;
- richer knowledge records, candidate lineage, profiler feedback, and broader instability diagnostics;
- CAKE IR, verifier, and cost-model integration.

## Required safety outcome

```text
D_REPO_ONLY = PASS
C_MOTHER_REPO_MODIFIED = NO
REMOTE_V100_USED = NO
HISTORICAL_CANDIDATES_MODIFIED = NO
```
