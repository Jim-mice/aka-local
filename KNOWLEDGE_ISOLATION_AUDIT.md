# Knowledge Isolation Audit

Target directories remain separated under `knowledge/targets/megatron_5be9626`.
CE forward scores are not used as RMSNorm, attention, or MoE scores; standalone
attention numbers are not real Megatron attention evidence; and the native
SequentialMLP fallback is not described as TE grouped GEMM. The project-level
registry links evidence by path instead of copying raw measurements across
targets.
