# Publication cleanup report

## Cleanup and exclusions

The public candidate excludes virtual environments, Python caches, runtime bundles, editor state, native/compiler output, raw profiler binaries, local gateway state, local event streams, and generated evaluator trees. A deletion attempt for `.venv`, `__pycache__`, and `.pytest_cache` was denied by the local execution safety policy; they remain local and ignored. The plaintext V100 secret file was removed. Root `_*.py`/`_*/` exploratory scripts, backups, and work directories remain local, ignored, and marked review-required rather than being guessed disposable.

Four generated RMSNorm JSON outputs of about 42.8 MB each and other malformed/raw evaluator transcripts are retained locally but excluded. Their compact contracts, candidate source, phase reports, canonical registries, and scientifically meaningful summaries remain candidates.

## Privacy and secret review

The pre-cleanup scan located a local credential file and a V100 configuration with private infrastructure details. The credential was removed; `v100.yaml` was replaced by `v100.example.yaml` with placeholders. Publication-candidate references to the known private endpoint, user, and absolute personal paths were replaced with `<REMOTE_HOST>`, `<REMOTE_USER>`, `<PROJECT_ROOT>`, `<LOCAL_USER_HOME>`, or `<REMOTE_HOME>`. Remote helper scripts use `AKA_V100_PASSWORD` at runtime and no longer read a tracked secret file.

The final candidate scan found no RFC1918 endpoint, known personal path, private key block, or authorization-header match. It found no known literal credential assignment; generic `password`/`token` references are API field names or environment-variable access and were manually reviewed in the central remote evaluator. Secret values are not recorded here.

## Third-party and licensing

`atrex-kernel-agent-win` is an ignored nested checkout with origin `https://github.com/alibaba/atrex-kernel-agent.git`. The ignored `lab/knowledge_sources` nested checkouts originate from GPU Mode lectures/resource-stream and Modal GPU Glossary. They are not vendored as aka-local work. The tracked target code references an external Megatron-LM checkout but does not publish that checkout.

No root `LICENSE` was found. This is `LICENSE_MISSING`; no license was chosen or added. No confirmed third-party licensing conflict was identified in the publication candidate, but downstream reuse terms remain unresolved until maintainers select a license.

## Documentation and verification

Added/updated: `README.md`, `docs/SETUP.md`, `SECURITY.md`, `config/environments/v100.example.yaml`, `.gitignore`, and the publication audit documents. The README links the canonical five-target summaries and explicitly distinguishes raw, stability-qualified, and promoted scores.

Lightweight checks used the existing local environment only: Python compile sanity for `lab`, `agent_backends`, `benchmarks`, `operators`, `ops`, `scripts`, `targets`, and `smoke`; JSON parsing for 1,010 publication-candidate JSON files; and `lab/cli.py --help`. They passed. No Agent campaign, GPU benchmark, remote V100 operation, or Megatron modification was performed. `git diff --cached --check` reports inherited trailing whitespace in historical reports/profiler text; this is non-functional and deliberately not mass-reformatted.
