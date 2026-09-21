# Recent Experiments

- remote-swiglu-v6: Bias SwiGLU direct2 regime [ACCEPT]
- remote-moe-v0: MoE Grouped MLP reference baseline [BASELINE]
- remote-moe-v1: MoE Grouped MLP Triton attempt [REJECT]
- remote-residual-v5: Residual RMSNorm flat row-batched [ACCEPT]
- aka-rmsnorm-v0: Standalone RMSNorm V0 eager reference [BASELINE]
- aka-rmsnorm-v1: Standalone RMSNorm V1 fused row reduction [PROMOTE]
- aka-rmsnorm-v2: Standalone RMSNorm V2 warp reduction [ACCEPT]
- aka-swiglu-v2b: Standalone SwiGLU V2b [ACCEPT]
- aka-swiglu-v3: Standalone SwiGLU V3 robustness [REJECT]