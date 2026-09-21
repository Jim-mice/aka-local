# Phase 14-B — Megatron SwiGLU Target Replay

## Verdict

**PASS for real target replay and observation.** No Megatron upstream file was modified. No replacement CUDA kernel was written. No Agent campaign was started.

## 1. Selected target

SwiGLU was selected because Phase 14-A classified its contract as CLEAR, reproducibility as MODERATE, and V100 feasibility as SUPPORTED_WITH_FALLBACK. The selected path is the real Megatron `MLP.forward` path with the non-TE, non-fused activation branch. This avoids claiming unavailable Transformer Engine or FP8 support while still executing the exact Megatron source implementation.

The replay boundary is broader than a newly invented `silu(A) * B` script: Megatron `MLP.forward` executes FC1, optional bias addition, gate/up split, SiLU-gated multiplication, and FC2. The initial replacement boundary is the activation sub-boundary; the observation harness executes the full MLP with TP=1 local linear adapters so the actual Megatron MLP path is exercised.

## 2. Exact source

Megatron checkout: `<LOCAL_USER_HOME>\projects\megatron-lm`

Verified HEAD: `5be9626709af2722333bf54797c954c09edeada3`

The upstream checkout remained clean. The isolated remote copy is `~/aka_targets/megatron-lm-5be9626/` and was separately verified at the same commit. Its working tree was cleaned by materializing the exact Git index after transfer; no upstream repository was modified.

## 3. Real call chain

```text
TransformerLayer._forward_mlp
  -> MLP.forward
  -> self.linear_fc1
  -> activation branch in megatron/core/transformer/mlp.py
       -> optional bias add
       -> torch.chunk(intermediate_parallel, 2, dim=-1)
       -> config.activation_func(gate) * (up + glu_linear_offset)
  -> self.linear_fc2
  -> (output, output_bias)
```

The exact source files are `megatron/core/transformer/mlp.py`, `megatron/core/fusions/fused_bias_swiglu.py`, and `megatron/core/extensions/transformer_engine.py`. Fused bias-SwiGLU and TE activation paths exist, but were deliberately excluded from this first compatibility replay and are recorded as open variants rather than silently substituted.

## 4. Dependency audit

| Dependency | V100 result | Classification |
|---|---|---|
| Python | `<REMOTE_HOME>/venvs/lerobot-act/bin/python`, Python 3.10 | AVAILABLE |
| PyTorch | `2.7.1+cu118`, CUDA 11.8 | AVAILABLE |
| CUDA GPU | Tesla V100-PCIE-16GB, CUDA available, sm_70 | AVAILABLE |
| Triton | importable on remote | AVAILABLE |
| Transformer Engine | not installed | OPTIONAL for selected path; BLOCKING for TE/FP8 variants |
| Apex | not installed | OPTIONAL for selected path; fallback warnings observed |
| Megatron exact source | isolated remote checkout | AVAILABLE |

Megatron emitted explicit warnings that TE/Apex were absent and Torch/local fallbacks were used. These warnings were retained as evidence; no dependency failure was hidden.

## 5. Minimal runtime configuration

The remote harness used:

```text
micro_batch=2
sequence_length=128
hidden_size=1024
ffn_hidden_size=4096
tensor_model_parallel_size=1
pipeline_model_parallel_size=1
gated_linear_unit=True
activation=F.silu
add_bias_linear=False
bias_activation_fusion=False
use_te_activation_func=False
```

No distributed process group was initialized because TP=1 and the selected local adapter path does not require a collective.

## 6–9. Shape, dtype, layout, fixture

The source-derived shape family used `[S,B,H]` cases `(16,1,1024)`, `(64,2,1024)`, and `(128,2,1024)`; FC1 local width is `2*I = 8192`, post-SwiGLU width is `I = 4096`, and FC2 returns `[S,B,1024]`. Dtype was `torch.float16`, supported by the V100/CUDA 11.8 stack and the selected PyTorch/Megatron path. This is not an FP8 or TE result.

Input and output were contiguous row-major with strides `[2048,1024,1]`. Fixture regeneration uses seed `14`; compact metadata is in `targets/megatron_5be9626/swiglu/fixtures/fixture_metadata.json`. Large tensors were not stored.

## 10. Semantic replay

The real Megatron class was imported from the exact checkout as `megatron.core.transformer.mlp.MLP`. The replay then independently performed the documented split/SiLU/multiply semantics and passed the same FC2 weights.

Observed result:

| metric | result |
|---|---:|
| shape | max absolute error | replay |
|---|---:|---|
| `[16,1,1024]` | `0.0` | PASS |
| `[64,2,1024]` | `0.0` | PASS |
| `[128,2,1024]` | `0.0` | PASS |

All cases used `torch.float16`; raw relative diagnostics are not meaningful where the exact reference element is zero, so the acceptance comparison used `allclose(atol=2e-3, rtol=2e-3)` and exact max-absolute evidence.

The exact contract is in `targets/megatron_5be9626/swiglu/replay_contract.json`.

## 11–12. Timing boundary and baseline

Setup outside timing: imports, module/config construction, parameter initialization, deterministic input generation, and warmup. The timed region contains repeated calls to the real `MLP.forward` only, with CUDA synchronization and CUDA events around 100 iterations. This is an observation baseline, not a candidate speedup score.

Measured V100 real Megatron MLP latency for this one shape: **approximately 626 µs per call**. A second run measured approximately 639 µs; the variation is retained as runtime evidence rather than promoted as an optimization claim.

## 13–15. REAL NSYS and static/runtime comparison

NSYS command:

```text
<REMOTE_HOME>/tools/nsys-2026.2.1/target-linux-x64/nsys profile --trace=cuda,nvtx,osrt --sample=none -o /tmp/aka_phase14b_swiglu_nsys <REMOTE_HOME>/venvs/lerobot-act/bin/python /tmp/aka_phase14b_swiglu.py
```

The actual GPU trace showed a multi-kernel graph, not one fused SwiGLU kernel:

| runtime kernel family | instances | total time |
|---|---:|---:|
| V100 FP16 GEMM (`volta_fp16_s884gemm_fp16_128x256_ldg8_f2f_tn`) | 224 | 56.66 ms |
| PyTorch SiLU elementwise | 112 | 4.98 ms |
| PyTorch add elementwise | 111 | 3.79 ms |
| cuBLAS split-K reduction | 112 | 2.89 ms |
| PyTorch multiply elementwise | 111 | 1.92 ms |

The trace confirms the static map: FC1/FC2 GEMMs plus separate activation kernels. It also adds runtime detail: this selected path is not a TE fused activation kernel, and the apparent “SwiGLU” region is a multi-kernel composition. The NSYS artifact summary is in `targets/megatron_5be9626/swiglu/nsys_stats.txt`.

## 16. Future replacement boundary

`targets/megatron_5be9626/swiglu/replacement_boundary.json` defines the future candidate boundary:

- inputs: `intermediate_parallel [S,B,2I/TP]` and optional bias;
- output: activated intermediate `[S,B,I/TP]`;
- preserve gate/up ordering and current CUDA stream semantics;
- support even last dimension and selected dtype;
- leave FC1, FC2, TP collectives, TE/FP8 scaling, dropout, and residual BDA outside scope.

Backward is not yet benchmarked or replaced; any future candidate must preserve gradients for intermediate and bias.

## 17. Isolated target namespace

Research evidence is isolated under:

- `targets/megatron_5be9626/swiglu/`
- `knowledge/targets/megatron_5be9626/swiglu/`

No standalone operator score or experience was copied into this namespace.

## 18. Standalone knowledge hashes

Project-scoped knowledge-tree hashes after Phase 14-B:

```text
rms_norm_v100_cuda              d9b911ca69f5f13c0c19fbc7387f58d28633cdfa9ad0edbc4026d142df1b559d
layer_norm_v100_cuda            06ab070462245dee4c93db3efb38d8d2204580ed32b592a4c865322e96b8d618
softmax_v100_cuda               642091538b7f0dac3c1e62bf46bd58e60526095238a2e81fcfa6184274490991
dense_attention_v100_cuda       317caefe1f1b095418655db2defcf60181afa7caa6a7e0b54c3b500445242ae9
causal_attention_v100_cuda      15876e288cb4522c540a35f3489fe5cb68ebc3efe8ca427255093d1e8056d897
dense_attention_backward_v100_cuda 6f55e1c677c2db0e542be530a4846426928f43c5369be25427ec5edba5002022
```

No files in those namespaces were changed by this phase.

## 19. Stable CLI

```powershell
.\.venv\Scripts\python.exe -m lab.cli target --target swiglu --action inspect
```

The adapter API supports `load_target_spec`, `prepare_inputs`, `run_megatron_reference`, `run_replay`, `compare_outputs`, and `benchmark_reference`. Replay/benchmark require a runtime where the exact Megatron import dependencies are available; the local Windows failure for missing Triton is reported as a dependency blocker, not replaced by a fake implementation.

## 20. Blockers and limitations

- The observed path uses TP=1 local linear adapters, not TE or distributed TP GEMMs. It proves the real Megatron MLP activation path and its surrounding call semantics, not every production backend.
- TE/Apex are absent on V100. TE activation, fused residual, and FP8 variants remain unvalidated.
- The current replay is forward-only. A future backward contract must capture and compare gradients before any kernel replacement.
- One representative shape was profiled with NSYS; three source-derived shapes were replayed semantically.
- No candidate kernel, score, incumbent, or Agent campaign was created.
