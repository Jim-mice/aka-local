# SwiGLU end-to-end blueprint

Status: `EXTERNAL_INTEGRATION_REQUIRED` for real nine-grid training evidence.

The source point is `MLP.forward` in
`megatron/core/transformer/mlp.py`, after TP-aware FC1 and before FC2. The
initial replacement interface is `bias_swiglu_impl` for
`use_te_activation_func=False`, `bias_activation_fusion=True`, SiLU gated
activation, and `per_token_scale=None`: input `[S,B,2I/TP]`, optional bias
`[2I/TP]`, and output `[S,B,I/TP]` on the current PyTorch CUDA stream. The TE
activation and weighted per-token paths are separate contracts, never silent
fallbacks. FC1, FC2, their collectives, and their parameter ownership remain
untouched.

Real shapes must be discovered at that source point by recording `S`, `B`,
`H`, `I`, `TP`, dtype, strides, bias/fusion flags, module identity, and config
hash. Tensor payloads and invented nine-grid dimensions are not part of the
manifest.

L0 compiles the candidate, validates its contract, compares forward and
backward with the equation `SiLU(gate+b0)*(up+b1)`, checks all captured shapes,
and measures stable activation-only latency. L1 installs the replacement in
the real MLP path, asserts a per-rank invocation marker and zero fallback,
compares complete MLP forward/backward, and records module time and peak
memory. L2 compares repeated baseline/candidate training windows for iteration
time, samples/s, tokens/s, peak memory, loss delta, and gradient sanity.

Artifacts are separated as `contract.json`, `shape_manifest.json`,
`provenance.json`, `l0/result.json`, `l1/result.json`, `l2/result.json`, and
`l2/raw_metrics.json`.
Each manifest entry carries the exact declared path and SHA-256; static
validation also requires the pinned Megatron commit, replacement marker, zero
fallback, and all three promotion axes. Bundle validation reads and hashes the
files, checks the L0-to-L2 verdict chain against promotion state, and requires
qualified L2 provenance to reference the hashed raw-metrics artifact. Each
result preserves raw metrics. The contract artifact must identify SwiGLU and
the pinned commit; the shape manifest must include non-empty captured
dimensions, dtype, strides, and config hash; provenance must identify the
runner, environment hash, commit, and measurement protocol. Empty placeholder
documents are rejected.
The contract artifact digest and one candidate SHA-256 are bound across L0,
L1 runner provenance, the bundle manifest, and the measurement-provenance
artifact used by L2; results from different candidates cannot be spliced into
one passing bundle.
`scripts/validate_operator_bundle.py --operator swiglu --bundle-root <path>`
provides the same validation as a read-only CLI. The repository fixture is
structural test data only; it is not external qualification evidence.
`KernelPromotion`, `IntegrationPromotion`,
and `SystemPromotion` are independent; only a qualified L2 result can accept
the system state.

The exact pretraining command cannot be authored locally without the nine-grid
model configuration, dataset, topology, credentials, warmup policy, and
measurement window. Those dependencies are recorded, not contacted.
`config/protocols/swiglu_end_to_end_template.json` is the handoff template:
external command/model/data/topology/run IDs/protocol evidence remain null or
empty and therefore cannot qualify, while required metrics, shape fields,
repeat count, fallback rule, and safety flags are already fixed.
