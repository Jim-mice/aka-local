# Five Real Megatron Target Campaign Summary

| Canonical target | Backend | Best valid evidence | Promotion state | Current blocker |
|---|---|---|---|---|
| MLP SwiGLU activation | Megatron/PyTorch activation | isolated fwd ~2.8x, bwd ~3.46x; current-stream integration | completed, integration-limited | surrounding GEMM/autograd overhead |
| Vocab-Parallel CE forward | native TP CE | Episode 7 2.4771907400291515x, CI95 [2.46543,2.48439] | promoted/frozen | none; frozen |
| Vocab-Parallel CE backward | native rank-local backward | correctness-valid raw B1 ~2.56x | not promoted | environment stability |
| WrappedTorchNorm RMSNorm | torch.nn.RMSNorm non-TE | correctness-valid raw R1/R2 ~1.64x | not promoted | candidate stability |
| Native DotProductAttention | native unfused core | A2 qualified 5.928x/1.167x/0.347x by S | not promoted | mixed scaling; A3 correctness failed |
| SequentialMLP expert compute | native per-expert MLP | M4 correct; raw 1.2793x/1.4816x/0.9573x | not promoted | two-empty reference stability |

TE paths were not substituted with synthetic fallbacks. The original historical
labels “Residual Add RMSNorm”, “Dense Fused Attention”, and “MoE Grouped GEMM”
must not be used as claims that those unavailable fused backends executed.
