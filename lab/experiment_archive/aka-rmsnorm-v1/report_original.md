# Standalone RMSNorm V1 fused row reduction

- ID: `aka-rmsnorm-v1`
- Operator: `rms_norm_train`
- Hardware: `rtx5060_laptop_sm120`
- Backend: `cuda_cpp`
- Decision: `PROMOTE`

## Evidence

56/56 correctness PASS; arithmetic mean speedup 7.202740x vs eager; one block per row, shared-memory tree reduction

## Provenance

{
  "source_type": "LEGACY_AKA_LOCAL",
  "source_path": "C:\\Users\\38154\\projects\\aka-local\\campaigns\\rms_norm\\backward\\episode_1\\decision.json",
  "source_commit": null,
  "canonical": true,
  "evidence_level": 3,
  "location": "local",
  "read_only": false
}
