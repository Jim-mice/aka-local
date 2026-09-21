# Bias SwiGLU direct2 regime

- ID: `remote-swiglu-v6`
- Operator: `bias_swiglu_train`
- Hardware: `biv150_corex`
- Backend: `corex_triton`
- Decision: `ACCEPT`

## Evidence

5-seed all-pass; score 2.940293; authoritative improvement 4.016 percent

## Provenance

{
  "source_type": "REMOTE_ATREX_CANONICAL",
  "source_path": "/private/atrex-megatron/campaigns/kernel_opt_bias_swiglu_train_triton_bi_v150_production/memory/v6.json",
  "source_commit": null,
  "canonical": true,
  "evidence_level": 3,
  "location": "remote_biv150",
  "read_only": true
}
