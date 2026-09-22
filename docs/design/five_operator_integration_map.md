# Five-operator Megatron integration map

This map targets Megatron-LM commit
`5be9626709af2722333bf54797c954c09edeada3`. The checkout was inspected
read-only. The five targets are fixed; this document is not an opportunity
search over other operators.

| Operator | Real source boundary | Replacement unit | Distributed concern |
|---|---|---|---|
| Dense Fused Attention | `Attention._run_core_attention` / `DotProductAttention.forward` | one declared core-attention backend and configuration | TP heads, optional context parallel, RNG |
| Vocab-parallel Cross Entropy | `LanguageModule.compute_language_model_loss` dispatch to `_VocabParallelCrossEntropy` | one declared full autograd variant with its exact collective trace | unfused MAX/SUM/SUM versus native-fused MAX/packed-SUM, plus vocab ownership |
| SwiGLU | `MLP.forward` between FC1 and FC2 | bias plus gated activation callable | TP intermediate width, optional sequence parallel |
| Residual Add RMSNorm | `self_attn_bda` output into `_forward_pre_mlp_layernorm` / `TENorm` | declared BDA-to-pre-MLP-norm bridge returning normalized output and post-add residual | residual dtype, dropout RNG, adjacent TP/SP ownership |
| MoE Grouped GEMM | `TEGroupedMLP.forward` expert compute | grouped FC1/FC2 with fixed routing metadata | EP dispatch/combine and TP expert weights |

Machine-readable contracts live in
`targets/megatron_5be9626/integration_contracts/`. Every contract records
source symbols, graph pattern, symbolic tensor shapes, training/backward
obligations, distributed context, an exact replacement boundary, fallback
detection, and metrics at operator, integration, and end-to-end levels.
`validate_integration_contract` rejects schema-version drift, unknown top-level
fields, missing or empty required evidence, unpinned commits, and absolute or
traversing source paths. All five checked-in contracts pass this validator.
`targets/megatron_5be9626/source_audit.json` records the SHA-256 of every source
file named by the five contracts plus operator-specific line anchors from the
read-only pinned checkout. This is static-source provenance, not runtime
dispatch or performance evidence.
`scripts/verify_megatron_source_audit.py --megatron-root <path>` performs a
read-only HEAD and digest recheck against that artifact.

The target names are research labels, not proof that a universal same-named
fused kernel exists. In particular, pinned Megatron's
`TEFusedResidualRMSNorm` forks a residual and normalizes; it does not itself
perform the preceding residual add. The target bridge therefore spans the
declared BDA output and following pre-MLP norm and must return both normalized
output and the post-add residual. MoE Grouped GEMM excludes routing and
communication but retains `permuted_probs` scaling inside expert compute.

Promotion requires independent `KernelPromotion`, `IntegrationPromotion`, and
`SystemPromotion` evidence. A standalone speedup cannot establish that the
replacement was invoked, preserved distributed behavior, or improved a full
training step.
