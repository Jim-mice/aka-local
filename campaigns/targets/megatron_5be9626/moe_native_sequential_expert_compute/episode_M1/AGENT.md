# Candidate agent

This directory contains one standalone candidate only. It does not modify Megatron upstream.

## Semantic ABI marker

`MEGATRON_NATIVE_SEQUENTIALMLP_EXPERT_COMPUTE_V1`

The exported entry point is:

```cpp
extern "C" void candidate(const half* tokens,
                           const int64_t* tokens_per_expert,
                           const half* probs,
                           const half* expert_fc1,
                           const half* expert_fc2,
                           half* output,
                           cudaStream_t stream);
```

Inputs are row-major `tokens[T,H]`, `tokens_per_expert[E]`, `probs[T]`,
`expert_fc1[E,2*I,H]`, and `expert_fc2[E,H,I]`; output is `output[T,H]`.
Tokens are assumed already permuted into contiguous local-expert slices.
The implementation preserves expert order, emits no rows for empty experts,
and launches on the supplied stream without synchronization.

No benchmark, framework edit, router, permutation, communication, TE, or
backward optimization is included.
