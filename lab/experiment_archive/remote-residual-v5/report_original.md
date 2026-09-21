# Residual RMSNorm flat row-batched

- ID: `remote-residual-v5`
- Operator: `residual_rmsnorm_train`
- Hardware: `biv150_corex`
- Backend: `corex_triton`
- Decision: `ACCEPT`

## Evidence

5 hidden workloads pass; 5-seed all-pass; score 19.75/19.81 vs V3 12.72; ~56 percent

## Provenance

{
  "source_type": "REMOTE_ATREX_CANONICAL",
  "source_path": "/private/atrex-megatron/campaigns/kernel_opt_residual_rmsnorm_train_triton_bi_v150_production/memory/v5.json",
  "source_commit": null,
  "canonical": true,
  "evidence_level": 3,
  "location": "remote_biv150",
  "read_only": true
}
