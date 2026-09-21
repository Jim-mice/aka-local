# aka-local

`aka-local` is a research workbench for examining candidate CUDA/operator implementations against real Megatron-LM execution boundaries. It is an evidence-preserving experiment repository, not a production Megatron-LM replacement and not a drop-in fused-kernel library.

## Scope and maturity

The repository preserves contracts, candidate source, validation policies, campaign manifests, compact evidence, and phase reports for five real targets:

1. Megatron MLP SwiGLU activation boundary;
2. Vocab-Parallel Cross Entropy (forward and rank-local backward);
3. non-TE `torch.nn.RMSNorm` / WrappedTorchNorm;
4. native `DotProductAttention` dense/no-mask/p=0 forward core; and
5. native `SequentialMLP` expert compute.

The canonical current state is in [FIVE_TARGET_REAL_MEGATRON_CAMPAIGN_SUMMARY.md](FIVE_TARGET_REAL_MEGATRON_CAMPAIGN_SUMMARY.md), [REAL_TARGET_CAMPAIGN_INDEX.md](REAL_TARGET_CAMPAIGN_INDEX.md), and [PHASE19A_FIVE_TARGET_CAMPAIGN_CLOSURE.md](PHASE19A_FIVE_TARGET_CAMPAIGN_CLOSURE.md). Those reports distinguish raw ratios, stability-qualified scores, and promoted scores; raw results are not promotion claims.

Historical labels such as “Residual Add RMSNorm”, “Dense Fused Attention”, and “MoE Grouped GEMM” do not imply that one corresponding real fused operator executed. Read the target mapping before comparing results.

## Architecture

`lab/` contains campaign orchestration, runtime checks, evaluator interfaces, and integrity policy. `targets/megatron_5be9626/` contains the target-specific contracts, candidates, diagnostics, and compact result artifacts. `operators/`, `ops/`, and `benchmarks/` contain supporting implementations and metadata. Root `PHASE*.md` reports and the canonical JSON registries provide the scientific narrative and machine-readable state.

An Agent/evaluator/promotion flow is intentionally gated: source identity and ABI contract are checked before delivery, correctness and benchmark scope are checked before environment/stability qualification, and only then can a score be promoted. Do not start campaigns casually: they may invoke CUDA builds or configured remote evaluators.

## Install and inspect

See [docs/SETUP.md](docs/SETUP.md). The safe starting commands are:

```powershell
py -3 -m json.tool .\five_target_campaign_state.json
py -3 -m json.tool .\real_target_campaign_index.json
.\lab.ps1 status
.\lab.ps1 list-ops
```

`run`, `evaluate`, and Agent commands are deliberately not examples here: they can use CUDA or remote infrastructure.

## Remote execution

Remote V100 evaluation is optional. Copy `config/environments/v100.example.yaml` to the ignored `v100.yaml`, replace `<REMOTE_HOST>`, `<REMOTE_USER>`, and path placeholders, then authenticate with SSH keys or an interactive local mechanism. Never commit passwords, tokens, private endpoint details, or local configuration.

## Repository layout

```text
config/                    public example and target evaluation configuration
lab/                       workbench runtime, policies, and CLI
targets/megatron_5be9626/  real-target contracts, source, and compact evidence
operators/, ops/           reusable operator/supporting source
campaigns/                 preserved campaign provenance where non-generated
PHASE*.md                  authoritative phase reports
docs/                      setup and publication guidance
```

Large raw profiler reports, virtual environments, runtime bundles, external source checkouts, machine-local configuration, and compiler products are intentionally excluded from Git. External source references are recorded in `lab/knowledge_sources/import_manifest.json` where applicable.
