# Public release report

## 1. Project and Git-root audit

- Original project root: `C:\Users\38154\projects\aka-local`
- `physical_project_root`: `C:\Users\38154\projects\aka-local`
- `local_dot_git_exists` before initialization: `False`
- `resolved_git_root` before initialization: none
- New independent Git root verified after `git init`: `C:\Users\38154\projects\aka-local`

No parent Git repository was mutated. Megatron-LM was not read for modification, changed, benchmarked, or contacted.

## 2. Cleanup inventory and local-only artifacts

The complete pre-cleanup inventory is in `PUBLICATION_CLEANUP_AUDIT.md`. Local-only ignored artifacts include `.venv`, caches, runtime bundles, root exploratory/backup scripts, raw profiler data, compiler products, generated evaluator trees, external source checkouts, and malformed historical JSON-like transcripts. A policy-denied attempt to remove `.venv` and caches left them locally available but ignored.

## 3. Preserved scientific evidence

Source, contracts, ABIs, candidate provenance, compact manifests/results, statistical policy, phase reports, canonical five-target state, and the five real target trees remain tracked. The root README prominently links the campaign summary, canonical index, and Phase 19-A closure. Raw ratios, stability-qualified scores, and promoted scores remain distinct.

## 4. Privacy, secrets, and attribution

A plaintext local credential file was removed and the real V100 configuration was replaced by `config/environments/v100.example.yaml`. Private endpoint/user/path references in publication candidates were replaced with neutral placeholders. The pre-commit scan found zero private-RFC1918 address matches, known personal-path matches, private-key matches, or authorization-header matches. No secret values are recorded.

Ignored external repositories are Atrex Kernel Agent (Alibaba), GPU Mode lecture/resource-stream material, Modal GPU Glossary, and a nested local campaign repository. Megatron-LM is an external referenced dependency, not a copied checkout. `LICENSE_MISSING`: no license was present and none was selected.

## 5. Documentation and tests

Added `README.md`, `docs/SETUP.md`, `SECURITY.md`, the public remote-config example, cleanup audit/report, and hardened `.gitignore`. Python compile sanity passed; 1,010 publication-candidate JSON files parsed successfully; and `lab/cli.py --help` passed. No Agent campaign, GPU benchmark, remote V100 operation, or Megatron modification was run.

## 6. Staging and publication state

- Publication commit: `ad5683fd3e7ae6a8abaefe0c96f1fa5e4465d9a8`
- Candidate at commit time: 1,792 files, about 10.38 MB
- Files over 50 MB: 0; files over 100 MB: 0
- GitHub CLI: unavailable (`gh` not found)
- Remote/origin: none created
- Public visibility verification: not applicable; no push occurred

Final publication state: **READY_FOR_GITHUB_AUTH**.
