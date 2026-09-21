# Phase 14-D — Megatron SwiGLU integration

## Result

Episode 2 was integrated through a reversible process-local hook in the exact
Megatron `MLP.forward` fused SwiGLU branch. The authoritative Megatron checkout
was not modified. Forward correctness and training gradients passed. The
activation-only score remains `2.7994219276258265x`; it is not an MLP or training
speedup.

## Provenance and boundary

Source commit: `5be9626709af2722333bf54797c954c09edeada3`.
Episode 2 candidate: `21e7995911f9af2d`. Optimization contract:
`2b05cc321bed0956`. The exact replaced region is only
`optional bias add -> first-half gate/second-half up split -> SiLU(gate) * (up + offset)`.
FC1, FC2, collectives, TE, FP8, dropout, and residual BDA remain outside.

The Phase 14-B ~626 us number was a complete `MLP.forward` measurement, so it
was not comparable to the Phase 14-C activation-only score. The integrated
measurements below include FC1, activation, and FC2 in both paths.

## Stream-safe integration

The original Episode 2 launcher used ordinary CUDA launch syntax and therefore
did not establish PyTorch current-stream behavior. The immutable candidate was
preserved. `integration_adapter.cu` adds `cudaStream_t` and launches with
`<<<..., 0, stream>>>`; Python passes
`torch.cuda.current_stream(device).cuda_stream` and performs no synchronization.
Adapter hash: `281fa3160db58a73`. Integration contract hash:
`00b8d9b26b2dc005`.

The runtime hook temporarily replaces the imported `megatron.core.transformer.mlp.bias_swiglu_impl`
symbol and restores it afterward. Thus the real `MLP.forward` executes real FC1,
the Episode 2 activation adapter, and real FC2; no upstream source edit is made.

## Correctness

Official shapes were `[16,1,1024]`, `[64,2,1024]`, `[128,2,1024]`, FP16, TP=1.
Final MLP output differences passed `atol=rtol=0.002`; maximum absolute
differences observed were `6.10e-5`, `1.22e-4`, and `1.22e-4`.

The custom autograd function saves the intermediate (and optional bias), uses
the exact gate/up ordering, and computes trusted PyTorch analytical backward.
At `[128,2,1024]`, input-gradient max absolute error was `1.4114e-4`; FC1 and
FC2 weight gradients had max absolute error `0.00390625`; all passed the
established FP16 gradient tolerance. Backward is not optimized.

## Benchmarks

For MLP forward, warmup and CUDA-event statistics were used with setup outside
timing. At `[128,2,1024]`, original mean was `631.392 us`, integrated mean was
`617.058 us`, giving `1.02323x`. This is the legitimate full-MLP forward
number from this run, not the activation-only result.

For forward+backward, optimizer work was excluded and gradients were reset per
iteration. Original mean was `2157.363 us`; integrated mean was `2348.851 us`,
giving `0.91848x`. The integrated backward is trusted reference PyTorch, so
this is a training-safe validation result, not an optimized training result.

## NSYS

Commands used remotely:

`nsys profile --trace=cuda,nvtx,osrt --sample=none -o phase14d_original ... integration_harness.py --mode original`

and the same command with `--mode profile` for the integrated path, followed by
`nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv`.

Original trace contained the FC1/FC2 Volta FP16 GEMMs plus separate SiLU,
add, and multiply kernels. Integrated trace contained the same GEMM family and
`swiglu_kernel(const __half*, const __half*, __half*, int, int, float)` with
1022 instances and about `32.0 ms` aggregate over the profiled repetitions.
This proves the candidate kernel executed inside the real MLP path. The
integrated trace still includes FC1/FC2; they were not replaced.

## Fallbacks and CLI

The wrapper is implemented to reject non-CUDA, non-FP16, non-contiguous, odd-width, wrong-device,
and unsupported-shape inputs with controlled `REJECT_INTEGRATION` errors. Negative
fixture execution was not completed after the user interruption; this remains a
follow-up validation item. The
stable commands are:

`python lab.py target --target swiglu --action inspect`

`python lab.py target --target swiglu --action validate-integration`

`python lab.py target --target swiglu --action benchmark-integration`

`python lab.py target --target swiglu --action profile-integration`

## Integrity and limitations

The target-specific manifest is
`campaigns/targets/megatron_5be9626/swiglu/integration_manifest.json`; lessons
are isolated under `knowledge/targets/megatron_5be9626/swiglu/integration/`.
Standalone operator namespaces were not changed. The original Megatron HEAD
remained the required commit and its working tree remained clean during the
integration runs. Phase 14-B/C artifacts remain the regression references.

This phase did not patch production Megatron, optimize backward, or claim a
training-step improvement. The current benchmark harness uses the official
shape set and TP=1 local linear adapters; broader Megatron configurations,
optional bias gradients, TE, FP8, and distributed execution remain unvalidated.
