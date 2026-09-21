# Phase 14-A — Megatron Operator Mapping

## Scope and provenance

This is reconnaissance only. No Megatron source was modified, no CUDA replacement was written, and no Agent campaign was started.

The inspected checkout is:

`<LOCAL_USER_HOME>\projects\megatron-lm`

Verified command result:

`git rev-parse HEAD = 5be9626709af2722333bf54797c954c09edeada3`

The checkout is clean. The upstream commit is the exact requested commit. Its Python/package metadata is in `pyproject.toml`, `setup.py`, and `uv.lock`; runtime optional dependencies include Transformer Engine, FlashAttention, Apex-adjacent integrations, and distributed PyTorch/NCCL paths. Availability was not assumed from source imports.

## Git-root audit

From `<PROJECT_ROOT>`, `git rev-parse --show-toplevel` returns `C:/Users/38154`, not the project directory. `git status -- .` therefore reports the project as an untracked subtree and can expose unrelated home-directory state. No `.git` directory was deleted, no reset/cleanup/init was run, and all inspection was project-scoped.

`_updated_evaluate.py` is an untracked 16 KB evaluator artifact with no attributable project Git history (`git log -- _updated_evaluate.py` produced no commit). It contains a Phase 13-C-style multi-operator evaluator and was not proven to be the authoritative remote evaluator. It was treated as a stale/derived duplicate and was not deleted.

## Source-tree overview

The relevant exact-commit boundaries are:

- `megatron/core/transformer/attention.py` and `dot_product_attention.py` — Transformer attention interface and local dot-product attention.
- `megatron/core/tensor_parallel/cross_entropy.py` and `core/fusions/fused_cross_entropy.py` — vocabulary-partitioned CE implementations.
- `megatron/core/transformer/mlp.py` and `core/fusions/fused_bias_swiglu.py` — dense MLP and SwiGLU activation boundaries.
- `megatron/core/extensions/transformer_engine.py` — TE builders, fused residual RMSNorm, TE activation and grouped-MLP paths.
- `megatron/core/transformer/moe/` — router, token dispatchers, MoE layer, experts, and grouped expert execution.
- `megatron/core/transformer/transformer_layer.py` — training graph composition and residual/BDA call sites.

## 1. Dense Fused Attention

### Real implementation

The target is not equivalent to aka-local `dense_attention_v100_cuda`. In this commit, `Attention.forward` in `core/transformer/attention.py` performs QKV projection, layout/rotary handling, invokes `self.core_attention`, and then applies the output projection. `Attention._run_core_attention` dispatches the configured core-attention module. The local implementation is `DotProductAttention.forward` in `core/transformer/dot_product_attention.py`; it imports `FusedScaleMaskSoftmax`. Layer specs can instead construct TE or other backend modules.

The implementation is therefore a composition of TP-aware GEMMs, QKV transforms, attention, and output projection. It is not one universal CUDA kernel. Configuration switches include the layer spec/`transformer_impl`, attention backend/configuration, checkpointing, flash-attention-related options, and TE availability. Flash-attention imports in `attention.py` include multiple modern variants; source presence is not evidence that a V100 can execute them.

### Forward and backward

```text
TransformerLayer.forward
  -> SelfAttention/Attention.forward
  -> linear_qkv
  -> QKV split/layout and rotary handling
  -> Attention._run_core_attention
  -> DotProductAttention.forward or selected external backend
  -> linear_proj
```

```text
loss.backward
  -> autograd for linear_qkv/linear_proj
  -> autograd Function or backend backward for core attention
  -> fused-softmax/attention backend backward
```

There is no single explicit Megatron backward kernel corresponding to the standalone aka-local contract. The exact backward boundary depends on the selected module spec and installed backend.

## 2. Vocab-parallel Cross Entropy

`core/tensor_parallel/cross_entropy.py` defines `_VocabParallelCrossEntropy` as a custom `torch.autograd.Function`. Inputs are partitioned logits and global targets. The final vocabulary dimension is local `V/TP`; source comments specify targets with `[sequence_length, micro_batch_size]` convention.

Forward performs local max, `MAX` all-reduce, local target-logit extraction/masking, `SUM` all-reduce for the target logit, exponentiation/local sum, `SUM` all-reduce for the denominator, and loss/normalization. The Function saves normalized probabilities, target mask, and masked target. Backward subtracts the target probability on the owning rank and writes the local partition gradient.

`language_module.py` selects among the ordinary tensor-parallel implementation, Megatron native fused implementation, and TE implementation according to `cross_entropy_loss_fusion` and `cross_entropy_fusion_impl`. This is a communication-plus-compute operator, not a safe single-GPU standalone kernel.

```text
LanguageModule.compute_language_model_loss
  -> vocab_parallel_cross_entropy / fused_vocab_parallel_cross_entropy / TE parallel CE
  -> TP MAX/SUM collectives + local math
  -> custom autograd backward + local vocab gradient
```

## 3. SwiGLU

`core/transformer/mlp.py:258` is the MLP forward boundary. For gated SiLU, FC1 produces interleaved gate/up channels. The unfused path chunks the local intermediate, computes `activation(gate) * (up + glu_linear_offset)`, and then calls FC2. With `bias_activation_fusion`, `bias_swiglu_impl` or `weighted_bias_swiglu_impl` from `core/fusions/fused_bias_swiglu.py` is selected. With `use_te_activation_func`, TE supplies the activation module; `core/extensions/transformer_engine.py` maps SiLU plus gated-linear-unit to `te.pytorch.ops.SwiGLU`.

```text
TransformerLayer._forward_mlp
  -> MLP.forward
  -> FC1 TP GEMM
  -> split gate/up
  -> SiLU(gate) * up, optionally fused with bias
  -> FC2 TP GEMM
```

Backward is autograd for the linears plus the explicit fused autograd implementation in `fused_bias_swiglu.py` or TE's activation backward. The realistic isolated boundary is bias+SwiGLU activation, not the entire MLP unless the TP linear contract is also captured.

## 4. Residual Add RMSNorm ambiguity

The exact source resolves the name as two related but distinct paths.

1. `core/fusions/fused_bias_dropout.py` implements `residual + dropout(x + bias)`. The source first combines `x_with_bias`, applies dropout, then adds residual. It does not implement `x + bias + dropout(residual)`.
2. `core/extensions/transformer_engine.py:626` defines `TEFusedResidualRMSNorm`. Its documented behavior is a residual fork (`MakeExtraOutput`) plus RMSNorm on the main path, returning `(normalized_output, residual)`. `build_norm(..., has_residual=True)` enables it only when `config.fused_residual_rmsnorm` is true and normalization is RMSNorm.
3. `TransformerLayer.forward` invokes BDA around attention/MLP outputs and carries residual explicitly.

Therefore “Residual Add RMSNorm” is not one unconditional operator in this commit. Target A—TE residual fork plus RMSNorm fusion—exists. Target B—`residual + dropout(x + bias)` followed by normalization—is a composition whose exact fusion depends on layer spec/config. The candidate contract must choose one path and state whether BDA is included.

## 5. MoE Grouped GEMM

`MoELayer.forward` in `core/transformer/moe/moe_layer.py` calls router/preprocess, token dispatcher, expert execution, and token combine. `router.py` produces top-k routing/probabilities. `token_dispatcher.py` performs permutation and, for relevant dispatchers, expert-parallel communication. `experts.py` contains `TEGroupedMLP` and `SequentialMLP` alternatives. The TE path provides grouped linear/GEMM; the fallback executes experts sequentially.

```text
TransformerLayer._forward_mlp
  -> MoELayer.forward
  -> Router.forward: logits -> top-k/probs/routing map
  -> dispatcher preprocess/dispatch: permutation + optional EP communication
  -> TEGroupedMLP or SequentialMLP
       -> expert FC1/grouped GEMM
       -> SwiGLU/activation
       -> expert FC2/grouped GEMM
  -> dispatcher combine: weighted unpermute + optional communication
```

Backward reverses the dispatcher/combine and differentiates through expert linears and activation; TE supplies grouped-linear backward when selected. Routing metadata, tokens-per-expert, expert count, top-k, and communication are part of the real execution context. Grouped GEMM alone is only the expert compute sub-boundary.

## Readiness and dependencies

| Target | contract clarity | reproducibility | external complexity | distributed dependency | V100 feasibility | boundary |
|---|---|---|---|---|---|---|
| Dense fused attention | PARTIAL | HARD | HIGH | OPTIONAL/CONFIGURED | FALLBACK | B/D |
| Vocab-parallel CE | CLEAR | HARD | MEDIUM | REQUIRED | FALLBACK | E |
| SwiGLU | CLEAR | MODERATE | MEDIUM | OPTIONAL TP | SUPPORTED_WITH_FALLBACK | B/C |
| Residual Add RMSNorm | AMBIGUOUS until path selected | HARD | HIGH | OPTIONAL/CONFIGURED | FALLBACK | B/D |
| MoE grouped GEMM | PARTIAL | HARD | HIGH | REQUIRED for EP configurations | FALLBACK | D/E |

### Dtypes

FP32, FP16, and BF16 are represented across the general paths. TE-specific FP8 and newer FP8/MXFP8 grouped/fused paths appear in source but require hardware and installed-version validation; they are not assumed V100-compatible. Residual streams can be explicitly FP32 via `fp32_residual_connection`. CE stability uses max subtraction and reductions; exact accumulation behavior varies by selected implementation.

### Symbolic shapes

| Target | symbolic shape |
|---|---|
| Attention | hidden `[S,B,H]`; Q/K/V `[S,B,heads_per_tp,head_dim]`; scores `[B*heads_per_tp,S,S]`; output `[S,B,H]` |
| Vocab CE | logits `[S*B,V/TP]`; target `[S*B]`; local gradient `[S*B,V/TP]` |
| SwiGLU | FC1 `[S*B,H] x [H,2I/TP]`; post-gate `[S*B,I/TP]`; FC2 to `[S*B,H]` |
| Residual RMSNorm | all streams `[S,B,H]`, normalization over final `H` |
| MoE | local tokens `T=S*B/(parallel partition)`; expert e input `[T_e,H]`; expert intermediate `[T_e,I_moe/TP]`; `E` experts, top-k assignments |

A later V100 study should instantiate shapes from a fixed Megatron configuration, for example `S`, micro-batch `B`, hidden `H`, heads, `I`, `V`, TP, `E`, and a recorded tokens-per-expert histogram. These values are intentionally not invented in this reconnaissance phase.

## Existing aka-local knowledge reuse

| Target | reuse classification |
|---|---|
| Dense fused attention | PARTIALLY REUSABLE: softmax/reduction concepts only; standalone scores/API do not transfer |
| Vocab-parallel CE | NOT DIRECTLY REUSABLE: standalone softmax knowledge omits TP collectives and target masking |
| SwiGLU | NOT DIRECTLY REUSABLE: no existing equivalent fused MLP boundary |
| Residual Add RMSNorm | PARTIALLY REUSABLE: RMSNorm reduction lessons; residual/BDA/TE semantics are new |
| MoE grouped GEMM | NOT DIRECTLY REUSABLE: existing dense attention/norm kernels do not model routing or grouped GEMM |

No findings were written into existing `knowledge/environments/v100_sm70/*` operator namespaces. Machine-readable specs are in `targets/megatron_5be9626/`.

## V100 feasibility and blockers

- A single-GPU V100 cannot validate TP/EP communication semantics. CE requires a multi-rank process group for faithful performance; MoE may additionally require expert parallel dispatch.
- TE must be installed at a version compatible with CUDA 11.8, V100, and the selected non-FP8 paths. TE FP8 and modern FlashAttention variants are not presumed available.
- Exact layer specs, `transformer_impl`, attention backend, fusion flags, TP/PP/CP/EP sizes, dtype, sequence length, and token routing distribution must be fixed before a reproducible campaign.
- For residual RMSNorm, the experiment must explicitly select TE fused residual RMSNorm or the BDA-plus-norm composition.
- For MoE, the campaign must decide whether it measures routing/communication plus expert compute or only grouped expert GEMM with routing metadata held fixed.

## Suggested next engineering experiments

1. Build a read-only import/config probe for the exact commit that records the selected layer specs and optional dependency versions without changing upstream.
2. Create a small multi-process TP V100 harness for vocab-parallel CE and capture MAX/SUM collective boundaries.
3. Reproduce a local non-FP8 SwiGLU activation boundary with fixed TP shapes and compare PyTorch versus Megatron fused bias-SwiGLU.
4. Probe TE residual RMSNorm availability on V100 and separately benchmark BDA and residual fork paths.
5. Start MoE with fixed routing metadata and SequentialMLP fallback before evaluating TE grouped GEMM or communication overlap.

These are proposals only; none was started in Phase 14-A.

## Acceptance status

- PASS — exact commit verified.
- PASS — five targets traced to source boundaries.
- PASS — forward/backward paths documented, including autograd/external cases.
- PASS — semantics, dtype classes, shapes, dependencies, and V100 limitations recorded.
- PASS — residual target ambiguity resolved from source evidence.
- PASS — five machine-readable specs created.
- PASS — no kernel optimization or Agent campaign started.
- PASS — existing aka-local knowledge namespaces untouched.
