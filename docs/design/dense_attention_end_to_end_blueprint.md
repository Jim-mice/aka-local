# Dense attention end-to-end blueprint

Status: `EXTERNAL_INTEGRATION_REQUIRED` for real training evidence.

The first declared boundary is the native dense/no-mask/dropout-zero core path
at `Attention._run_core_attention -> DotProductAttention.forward`. QKV and
output projections, rotary embedding, RNG, and collectives are excluded. Shape
capture records sequence lengths, batch, heads per TP rank, head dimension,
mask type, dropout, dtype, strides, and selected backend.

L0 checks QK-scale-softmax-PV against an FP32 oracle. L1 asserts backend
identity and invocation markers, forbids silent fallback, and compares full
attention forward plus dQ/dK/dV. L2 uses the same system metrics and policy as
SwiGLU. Any masked, dropout, context-parallel, inference, or alternate backend
variant requires its own contract rather than inheriting evidence from the
initial boundary.

The artifact manifest uses the same fail-closed path/SHA-256, pinned-commit,
replacement-marker, zero-fallback, and three-axis promotion validation as the
SwiGLU blueprint. Its result bundle likewise binds L0/L1/L2 verdict content to
promotion state and requires qualified L2 run provenance to reference the
hashed `l2/raw_metrics.json` artifact. The same non-empty contract identity,
shape-capture, and measurement-provenance schemas apply.
Contract and candidate SHA-256 identities use the same L0/L1/manifest/L2
provenance binding as SwiGLU.

`config/protocols/dense_fused_attention_end_to_end_template.json` mirrors the
SwiGLU handoff: the real command/model/data/topology and qualification evidence
remain unset, while required shapes, metrics (including samples/s), repeats,
fallback behavior, and safety flags are fixed.
