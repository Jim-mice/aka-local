# Standalone RMSNorm V2 warp reduction

- ID: `aka-rmsnorm-v2`
- Operator: `rms_norm_train`
- Hardware: `rtx5060_laptop_sm120`
- Backend: `cuda_cpp`
- Decision: `ACCEPT`

## Evidence

56/56 correctness PASS; repeated ABBA arithmetic mean ~1.074x; 50/56 winning shape means; shared memory 2048B to 1056B

## Provenance

{
  "source_type": "LEGACY_AKA_LOCAL",
  "source_path": "C:\\Users\\38154\\projects\\aka-local\\campaigns\\rms_norm\\episode_2\\decision.json",
  "source_commit": null,
  "canonical": true,
  "evidence_level": 3,
  "location": "local",
  "read_only": false
}
