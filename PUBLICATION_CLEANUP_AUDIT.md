# Publication cleanup audit

Date: 2026-09-21. Scope is limited to `<PROJECT_ROOT>`.

## Git-root audit

| Field | Observed value |
|---|---|
| physical_project_root | `<PROJECT_ROOT>` |
| local_dot_git_exists | `False` before publication initialization |
| resolved_git_root | none returned by `git rev-parse --show-toplevel` |

Nested repositories were found at `atrex-kernel-agent-win`, `lab/campaigns/rms_norm_train__rtx5060_sm120__cuda_cpp`, and the three checkouts under `lab/knowledge_sources`. They are not treated as content of the new repository.

## Inventory before disposable-artifact deletion

| Class | Finding | Disposition |
|---|---|---|
| Source, CUDA, Python, PowerShell | `lab`, `targets`, `operators`, `ops`, scripts and metadata | KEEP |
| Phase reports and canonical state | root `PHASE*.md`, campaign index/state JSON | KEEP |
| Campaign trees | ~238.5 MB, mixed provenance and compiler products | REVIEW_REQUIRED; ignore generated native products only |
| Lab tree | ~194.6 MB, includes nested external/materialized campaign data | KEEP source; IGNORE nested repositories/local runtime data |
| Runtime bundle | ~380.1 MB | IGNORE (reproducible local runtime) |
| `.venv` | ~3.3 GB | DELETE_SAFE after documenting reconstruction |
| Python/test caches | `__pycache__`, `.pytest_cache` | DELETE_SAFE |
| Root underscore scripts/backups/work trees | many one-off remote/probe/rewrite scripts | REVIEW_REQUIRED; retain locally but IGNORE from publication |
| Raw profiler artifacts | `.ncu-repz` observed; other raw profiler extensions scanned | IGNORE |
| Native build products | `*.o` under campaign/evaluation trees | IGNORE |
| External checkout | Atrex clone, GPU Mode/Modal clones | IGNORE; retain upstream attribution |
| Credentials/config | `.v100_secret`, `v100.yaml` | DELETE_SAFE / redact; replace with public example |

## Large-file audit

Files over 50 MB were reproducible runtime/external-checkout material (including a ~284 MB runtime executable and transient nested-repository packs). Files between 10 and 50 MB included local generated runtime products and campaign evaluation outputs, including ~42.9 MB artifacts. No such artifact is designated for normal Git publication. Compact reports, contracts, source, and small result summaries are preserved.

## Privacy and licensing pre-check

The pre-cleanup scan found a plaintext local secret file and a V100 configuration containing a private endpoint, user identifier, and private paths. Secret values are intentionally omitted from this audit. The configuration was replaced by a sanitized example; the secret file was removed.

No root `LICENSE` was found. This is `LICENSE_MISSING`; no license is selected by this cleanup.
