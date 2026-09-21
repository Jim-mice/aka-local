# Phase 18-A — Megatron MoE Expert Compute Qualification

Status: `QUALIFIED_FOR_AGENT_CAMPAIGN` for a local, fixed-routing `SequentialMLP` expert-compute boundary; TE grouped GEMM is `DEPENDENCY_BLOCKED`.

## Source truth

At commit `5be9626709af2722333bf54797c954c09edeada3`, the path is:

`TransformerLayer._forward_mlp → MoELayer.forward → router → dispatcher preprocess/dispatch → experts.forward → dispatcher combine`.

`megatron/core/transformer/moe/experts.py` contains both `TEGroupedMLP` and `SequentialMLP`. `SequentialMLP.forward` converts `tokens_per_expert` to a split list, splits the permuted token tensor, invokes each local expert sequentially, and concatenates outputs. Each expert `MLP.forward` performs FC1, activation/SwiGLU, and FC2. Routing, permutation, optional expert-parallel communication, and combine are not grouped GEMM.

## Dependency audit

Transformer Engine grouped-linear execution is not available in the established V100 environment and was not installed or faked. TE grouped GEMM is therefore dependency-blocked. The source-valid PyTorch/Megatron `SequentialMLP` fallback is available in principle and is the canonical replay target for a local qualification. EP/TP communication is outside the isolated expert-compute boundary and must not be accidentally included on only one side of a future comparison.

## Canonical boundary

Canonical name: `Megatron Native SequentialMLP Expert Compute`.

The narrow boundary is fixed-routing local expert execution with inputs `[tokens,H]`, `tokens_per_expert`, expert FC1/FC2 weights and activation configuration, and output `[tokens,H]`. Router logits/top-k, token permutation, all-to-all, combine, and auxiliary loss are excluded. Valid distributions must include balanced, mildly imbalanced, highly imbalanced, and zero-token expert cases where the selected dispatcher permits them.

## Semantics and qualification status

The real source semantics are two expert linear operations with the configured gated SiLU activation between them, executed sequentially per local expert. Backward differentiates the same expert modules. The real V100 replay constructed `SequentialMLP` with E=4, H=64, MoE FFN=128, FP16, top-k=1, TP=1, EP=1; balanced `[4,4,4,4]`, moderate `[1,3,5,7]`, and zero-token `[0,0,8,8]` distributions all passed finite forward and input-gradient checks. Independent FP32 per-expert matmul/activation/oracle max absolute errors were 2.84e-6, 3.05e-6, and 3.91e-6 respectively. Contract hashes are recorded in `targets/megatron_5be9626/moe_native_sequential_expert_compute/qualification_manifest.json`.

## Headroom and limitations

Qualification timing means were 4144.2 us balanced, 4068.4 us moderately imbalanced, and 3689.2 us with two empty experts. REAL NSYS qdrep artifacts show many small expert GEMM/activation/cat operations rather than grouped TE execution: balanced had 3 kernel families/13 instances, moderate imbalance 6/13, and two-empty 3/7. No NCCL was inside the isolated boundary. Diagnosis: `SEQUENTIAL_EXPERT_GEMMS`, `MANY_SMALL_GEMMS`, `LAUNCH_OVERHEAD`, `ACTIVATION_KERNEL_OVERHEAD`, and `GROUPED_GEMM_DEPENDENCY_BLOCKED`. The real graph and clear expert-loop overhead provide sufficient headroom for a bounded Agent campaign.

## Replay, contracts, and future campaign

Replay contract hash: `e73232a4e11649d6519785c79ab4a823eddbd5588e2e318edbdae3abb99ecaec`.
Performance contract hash: `3086f89790979dac0e1a1918b9fb51953ac292820db6f14bbf414312858bdaef`.
Research candidate boundary and ABI are in the canonical target directory. The frozen future baseline policy is 3 independent invocations × 3 blocks × 10 samples, CV ≤ 0.20, with raw retention and paired ABBA for candidate comparison. The reference baseline completed this policy with 90 samples per distribution and passed.

The real NSYS qdrep artifacts are preserved in `moe_native_sequential_expert_compute/`; kernel summaries show the balanced path has 13 kernel instances, moderate imbalance 13, and two empty experts 7. The graph includes CUTLASS/cuBLAS-like expert GEMM families, `triton_poi_fused_mul_silu_0`, and concatenation; it is not TE grouped GEMM. This supports `SEQUENTIAL_EXPERT_GEMMS`, `MANY_SMALL_GEMMS`, `LAUNCH_OVERHEAD`, `ACTIVATION_KERNEL_OVERHEAD`, and `IMBALANCE_SENSITIVITY` as structural diagnoses only.

Phase 18-B was authorized. M1 was generated as a real Agent episode and is `REJECT_CONTRACT`: it exports `candidate` rather than the frozen required semantic entry point. M2 was then generated from that precise failure. M2 exports the required symbol but lacks the required semantic target marker, so it is also `REJECT_CONTRACT`; it was not manually repaired, compiled, benchmarked, or promoted. The bounded M1/M2 budget is exhausted without a valid candidate. No MoE incumbent exists.

Attention A2 and all prior CE/RMSNorm histories remain isolated and unchanged.
