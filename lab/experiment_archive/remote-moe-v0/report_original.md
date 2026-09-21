# MoE Grouped MLP reference baseline

- ID: `remote-moe-v0`
- Operator: `moe_grouped_mlp_train`
- Hardware: `biv150_corex`
- Backend: `corex_triton`
- Decision: `BASELINE`

## Evidence

11 shapes; correctness PASS

## Provenance

{
  "source_type": "REMOTE_ATREX_CANONICAL",
  "source_path": "/private/atrex-megatron/campaigns/kernel_opt_moe_grouped_mlp_train_triton_bi_v150_production/memory/v0.json",
  "source_commit": null,
  "canonical": true,
  "evidence_level": 2,
  "location": "remote_biv150",
  "read_only": true
}
