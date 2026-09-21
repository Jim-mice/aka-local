# RMSNorm episode 2 candidate

- Target: RTX 5060, `sm_120`; operator: BF16 RMSNorm forward.
- Selected exactly one unresolved frontier direction: `warp_reduce_shared_finalize`.
- Mechanism: one 256-thread block per row; each warp reduces its register sum with `__shfl_down_sync`; eight warp totals are finalized through one warp; normalization and BF16 output remain fused in the same kernel.
- Controlled boundary: compared with the incumbent, only the reduction implementation changed. The one-block-per-row mapping, two row traversals, float32 accumulation, epsilon handling, weight access, and output conversion are retained.
- Provenance: incumbent is `C:/Users/38154/projects/aka-local/ops/rms_norm_v1/candidate.py`; selection is based on the official frontier and `knowledge/experience/episode_1.json`.
- No CUDA, evaluator, benchmark, or profile was run for this episode. Evaluation remains pending.
