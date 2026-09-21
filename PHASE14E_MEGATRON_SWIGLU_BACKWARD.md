# Phase 14-E — Megatron SwiGLU backward

## Verdict

The saved-PyTorch SwiGLU backward boundary was frozen, independently checked,
optimized by a separate real Agent campaign, and integrated through a
current-stream adapter. Backward-boundary correctness and NSYS execution pass.
The optimized backward did not produce a net whole-MLP training gain in this
run: full MLP forward+backward measured `0.9071x`. FC1/FC2 remain unchanged.

## 1. Real boundary and equations

The Phase 14-D wrapper saves the contiguous FP16 FC1 intermediate and optional
bias. Official Megatron uses `add_bias_linear=false`, so bias is null for this
contract. `grad_output` is received by autograd. The actual split is last-axis:
`gate = z[..., :width/2]`, `up = z[..., width/2:]`, and
`out = SiLU(gate) * (up + offset)`, with offset zero in official shapes.

The replacement computes:

`grad_up = grad_output * SiLU(gate)`

`grad_gate = grad_output * (up + offset) * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))`

and writes `[grad_gate, grad_up]` back into `grad_intermediate`.
Bias gradient is explicitly outside this official candidate ABI because the real
configuration has no FC1 bias.

## 2. Contract and fixtures

Backward contract: `a7437b0c42289669`.
Forward contract: `2b05cc321bed0956`.
Integration contract: `00b8d9b26b2dc005`.
Official shapes: `[16,1,1024]`, `[64,2,1024]`, `[128,2,1024]`; rows are `S*B`,
width is `8192`, input/output dtype is FP16, contiguous row-major, TP=1.
The ABI is recorded in `targets/megatron_5be9626/swiglu/backward_optimization_contract.json`.

Fixtures use deterministic seeds, saved FC1 intermediate, and deterministic
grad_output. No FC1/FC2 or forward work is inside the backward timing boundary.

## 3. Correctness oracle and finite difference

The saved-P analytical equations were checked against the independent PyTorch
autograd derivative for the same boundary on all official shapes. Episode 2
passed `torch.allclose(atol=0.002, rtol=0.002)` for every grad element. Raw
maximum absolute errors were `0.0078125`, `0.015625`, and `0.01171875`; these
are reported because FP16 relative tolerance is part of the established oracle.

A FP32 finite-difference sanity test on three selected input elements produced
maximum absolute error `8.94e-6` and passed. Evidence is in
`targets/megatron_5be9626/swiglu/backward_sanity.py`.

## 4. Fair backward baseline

Inside timing: only the saved-intermediate plus grad_output PyTorch analytical
backward equations to grad_intermediate. Outside timing: fixture generation,
forward, GEMMs, model setup, and warmup. Five samples of 100 CUDA-event
iterations were used per shape.

Episode 2 baseline/candidate results:

| Shape | Baseline us | Candidate us | Speedup |
|---|---:|---:|---:|
| 16x1x1024 | 177.19 | 58.42 | 3.033x |
| 64x2x1024 | 178.28 | 57.72 | 3.089x |
| 128x2x1024 | 249.11 | 56.44 | 4.413x |

Geometric mean: `3.4578303034730578x`. Baseline hash is
`b0554f3fc5ada9fb` and the backward incumbent is isolated under
`campaigns/targets/megatron_5be9626/swiglu_backward/`.

## 5. Agent episodes and feedback loop

- Episode 1: real Codex candidate; compile failed. Preserved as `REJECT_COMPILE`.
- Episode 2: compile, all gradients, benchmark, and profile passed; incumbent.
- Episode 3: real Codex candidate; compile failed. Preserved as `REJECT_COMPILE`.

Episode 3 prompt contains the required current-performance, real-profile,
diagnosis, recommendations, worked/failed strategy, and next-search sections.
No candidate was manually repaired.

## 6. NSYS and diagnosis

Command:

`nsys profile --trace=cuda,nvtx,osrt --sample=none -o phase14e_backward python backward_remote_evaluate.py --candidate ...`

The reference graph showed multiple PyTorch elementwise kernels: multiply,
sigmoid, products, and concatenation. The candidate trace showed
`swiglu_backward_kernel` with 1533 launches and approximately 46.8 ms aggregate
over the profiled repetitions. The evidence supports `MULTI_KERNEL_OVERHEAD`
as the optimization diagnosis; no unsupported memory-bound claim is made.

## 7. Replay and integration

Deterministic replay passed all checks: commit, contract hash, candidate hash,
shape set, FP16 dtype, and saved-P baseline mode. Replay evidence is
`episode_2/replay_result.json`.

The stream-aware adapter is
`backward_integration_adapter.cu`, hash `54a5fc0b67c3259a`. It launches on
`torch.cuda.current_stream(device).cuda_stream` without synchronization. The
Phase 14-D autograd wrapper now dispatches to this adapter when bias is absent;
the trusted PyTorch path remains available for bias cases outside the official
contract.

## 8. Full Megatron correctness and performance

The real `megatron.core.transformer.mlp.MLP.forward` sidecar path still passed
all three forward shapes. After installing optimized backward, input-gradient
max absolute error was `1.3733e-4`; FC1 and FC2 weight-gradient max absolute
errors were `0.00390625`; all passed established FP16 tolerances.

The prior full MLP forward result remains approximately `1.023x`; it was not
silently overwritten. The rerun full forward+backward result with optimized
backward was:

- original mean: `2133.61 us`
- integrated mean: `2352.13 us`
- speedup: `0.90710x`

Thus activation-forward, activation-backward, MLP-forward, and full training
metrics remain separate. The backward boundary is faster, but wrapper/autograd
and surrounding backward work dominate this small TP=1 MLP configuration.

## 9. Training NSYS

The training profile was captured with the same NSYS command and `--mode train`.
It contains both `swiglu_kernel` for forward and `k(const __half*, const
__half*, __half*, int, int, __half)` for optimized backward, alongside the
unchanged FC1/FC2 GEMMs and surrounding PyTorch backward kernels. This proves
both optimized subgraph directions execute in the real Megatron-derived graph.

## 10. Negative integration tests

Evidence: `targets/megatron_5be9626/swiglu/backward/negative_tests.json`.
Wrong dtype, non-contiguous input, unsupported shape, and CPU input all produced
controlled rejection before launch. Each recorded `candidate_launched=false`.
There is no runtime metadata activation gate for contract mismatch, so that case
is not applicable rather than simulated.

## 11. Regressions and integrity

Phase 14-B replay remained PASS. Phase 14-C Episode 2 forward deterministic
replay remained PASS. Phase 14-D forward and gradient regression remained PASS.
The authoritative Megatron checkout still has HEAD
`5be9626709af2722333bf54797c954c09edeada3` and an empty short status.

No Megatron source was modified. No FC1/FC2 optimization or new forward
campaign was started. Backward knowledge is isolated under
`knowledge/targets/megatron_5be9626/swiglu/backward/`.

## Remaining limitations

The backward contract covers the official bias-free configuration only; bias
gradient and TE/FP8/distributed paths are not optimized. The full MLP training
result remains below 1x, so this phase demonstrates a complete executable
forward/backward replacement boundary but not a net training speedup for this
configuration. Broader Megatron configurations require separate contracts and
fixtures.
