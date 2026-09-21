# Phase 16-B — Megatron non-TE Torch RMSNorm forward campaign

## Status

`STABILITY_BLOCKED_R1_R2`.  This phase kept the qualified target narrow: `WrappedTorchNorm -> torch.nn.RMSNorm` forward only.  It did not include BDA, residual addition, dropout, Transformer Engine, NCCL, or RMSNorm backward optimization.  No incumbent, candidate promotion NSYS, deterministic incumbent replay, R3, or integration was created.

## Frozen inputs

- Replay contract: `5588960649248c710c9ae57f0b613ee4e881564aa8bf6a4361b2f56276bcecb6`
- Optimization contract: `d24ea7a30fb6a91bc1958a955d140992e77e68dad78faa48158f2b907b722731`
- Performance contract: `86a7faf7a7cb02c0e4a2fbf8599e0f60d353079e607cb8c2900668e3d79d95b6`
- Statistical policy: `df84717869eaf2bd81308d5434ef83de1496e1a2d8f8822db4d418784c22fa26`

The active ABI is out-of-place FP16 `rmsnorm_forward_fp16_stream(const __half* x, const __half* weight, __half* output, int64_t rows, int64_t hidden_size, float epsilon, cudaStream_t stream)`.  Inputs are contiguous/read-only, output is preallocated/non-overlapping, accumulation is FP32-equivalent, final output is FP16, epsilon is `1e-5`, and H=1024 specialization is explicitly allowed only for the three official shapes.

## R1/R2 identity and hard gates

| Episode | Candidate SHA256 | Preflight/build | Official forward | Edge | Reference-backward sidecar | Negative adapter tests |
|---|---|---|---|---|---|---|
| R1 | `f8c3eee376b002e1b5296fda64f00d8a6a5d5a569a87dc50462b86dc3977b117` | PASS/PASS | PASS, max abs 0.001953125 | PASS | PASS | PASS |
| R2 | `b8d3accbf198752c6793f18f6de89fdf77414b5a816b9e4ae5361f824e8891eb` | PASS/PASS | PASS, max abs 0.001953125 | PASS | PASS | PASS |

The edge set covered zero, constant, positive, negative, small, and large input.  The trusted sidecar intentionally retains PyTorch/reference backward and validates x/weight gradients against that trusted path; no custom backward is claimed.  The adapter rejects wrong dtype, noncontiguous x/weight, and invalid output shape before candidate launch.  The original hard-gate files are preserved; `hard_gates_v2.json` is the corrected validator version because its initial weight view was accidentally contiguous and it compared FP16 reference gradients directly to a separate FP32 oracle.

## Shared scope and active baseline

Reference and candidates use the same deterministic fixture, preallocated output, epsilon, current stream, warmup (`5`), CUDA-event boundary, synchronization, measurement count, and raw schema.  The timed region is only output materialization from preallocated x/weight/output.  Construction, allocations, startup, backward, and unrelated synchronization are outside both paths.

The active reference baseline is [baseline_manifest.json](C:/Users/38154/projects/aka-local/targets/megatron_5be9626/residual_rmsnorm/baseline_manifest.json), SHA256 `f654ea7ef4433d55b7df0490f72680d7dbfe5fb656577a459e9bdad554d8dee6`.  It contains three independent invocations, three blocks, and ten measurements per block/config; all raw data and read-only environment provenance are in `baseline_raw.json`.  Reference CVs were below the frozen 0.20 gate.

## Paired ABBA results

Each episode used `reference, candidate, candidate, reference` with 180 raw samples per implementation/config.  Valid slow samples were retained.

| Episode | Shape | Reference us / CV | Candidate us / CV | Raw speedup | Gate |
|---|---|---:|---:|---:|---|
| R1 | 16x1x1024 | 219.518 / .0846 | 134.025 / .0787 | 1.6379x | pass |
| R1 | 64x2x1024 | 216.519 / .0873 | 137.289 / .2174 | 1.5771x | **candidate fail** |
| R1 | 128x2x1024 | 225.706 / .1016 | 130.669 / .0480 | 1.7273x | pass |
| R2 | 16x1x1024 | 222.919 / .1110 | 132.392 / .0610 | 1.6838x | pass |
| R2 | 64x2x1024 | 217.226 / .1708 | 133.911 / .2063 | 1.6222x | **candidate fail** |
| R2 | 128x2x1024 | 220.268 / .1500 | 136.062 / .1935 | 1.6189x | pass |

R1 raw geometric mean was `1.64629x`; R2 raw geometric mean was `1.64134x`.  Neither is publishable because the predeclared maximum candidate CV is `0.20` and both fail it at 64x2x1024.  No CI, promotion score, or incumbent was produced.

## Profile and subsequent-Agent policy

Phase 16-A reference NSYS remains the only valid profile: approximately eight ATen kernels/call, zero NCCL.  Candidate promotion NSYS is intentionally absent because neither candidate is stability-qualified.  The required profile-to-R3 evidence chain therefore does not exist; R3 was not generated.  R4/R5 are likewise unjustified.

## Exclusions, integrity, and next condition

TE residual-fork RMSNorm remains `DEPENDENCY_BLOCKED`; BDA remains `SEPARATE_RNG_SENSITIVE_TARGET`.  CE forward remains `FORWARD_CAMPAIGN_FROZEN`; CE backward remains `ENVIRONMENT_STABILITY_BLOCKED`.  Megatron remained at `5be9626709af2722333bf54797c954c09edeada3` with clean status.

The only blocker is candidate timing stability under the frozen policy.  Future continuation requires an explicit new phase decision; this phase must not weaken CV, delete slow measurements, or generate R3 without a valid stability-qualified profile.
