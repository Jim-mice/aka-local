# Techniques

- biv150-swiglu-direct-loop: In the BI-V150 Bias SwiGLU campaign, expanding the single-launch direct serial row-loop regime to medium T regressed the medium-T shape and was reverted.
- swiglu-v3-noise-reject: On the standalone RTX5060 SwiGLU workload, V3's initial positive arithmetic result did not survive repeated ABBA robustness; V2b remained incumbent.
- biv150-corex-masked-2d-tile: A masked 2D row-tile implementation produced OOB corruption on the BI-V150 CoreX/Triton backend in the Residual RMSNorm campaign.
- rmsnorm-v2-local-observation: Standalone RMSNorm V2 reduced shared memory from 2048B to 1056B while retaining 30 registers/thread; repeated local ABBA recorded about 1.074 arithmetic speedup.
- coverage-open-questions: How do the same training-operator mechanisms behave on V100, BI-V150, and RTX5060 under matched contracts and workloads?