# Phase 15-B.1 — Agent Reliability Hardening

## Verdict

**BLOCKED after five new real Agent episodes.** The reliability layer now catches the two Phase 15-B failure classes before remote compilation, and one new Agent candidate compiled successfully. However, none of the five new candidates reached TP=1 correctness; therefore no candidate reached TP=2 correctness, benchmark, NSYS, or incumbent eligibility. No Agent source was manually repaired.

## 1. Phase 15-B root-cause audit

| Episodes | Failure | Root cause | Static prevention | Prompt deficiency | Validator deficiency | Toolchain |
|---|---|---|---|---|---|---|
| 1–3 | compile | Agent emitted `CUDART_INF_F` rejected by remote CUDA 11.8 nvcc | yes | CUDA restriction not prominent | no forbidden-token guard | verified toolchain fact |
| 4–5 | contract/runtime ABI | Agent omitted `local_max_fp16_stream` and companion stream symbol | yes | ABI was prose/preferred rather than canonical verbatim prototypes | no symbol preflight | no |

Episodes 1–3 had the contract marker and described stream-aware behavior, but the prompt did not state the exact prototype as a non-negotiable ABI and had no CUDA compatibility section. Episodes 4–5 compiled, proving the compiler issue was fixed by the Agent, but the evaluator attempted to resolve a missing symbol only after remote launch. The failure was preventable before nvcc and before NCCL.

The compile/ABI diagnostics were not structured into the original next prompt. Phase 15-B.1 adds explicit diagnostic categories and feeds concise evidence into later prompts.

## 2. Frozen scientific contract

The Phase 15-B optimization contract remains unchanged: `112959ca63020e9b`. The TP=2 baseline remains unchanged: `f11fd1fc6846d6f2`. No new scientific score era or semantic contract was created.

The collective invariant remains exactly:

```text
MAX all-reduce
SUM all-reduce(predicted target logit)
SUM all-reduce(exp denominator)
```

Backward remains local-only at this Megatron commit.

## 3. Canonical ABI

Machine-readable ABI: `targets/megatron_5be9626/vocab_parallel_cross_entropy/required_abi.json`.

Required prototypes, included verbatim in the new prompts:

```cpp
extern "C" cudaError_t local_max_fp16_stream(const half* logits, float* row_max, int rows, int local_vocab, cudaStream_t stream);
extern "C" cudaError_t local_prepare_fp16_stream(const half* logits, const int64_t* targets, const float* global_max, float* predicted_local, float* denominator_local, float* exp_values, unsigned char* target_mask, int64_t* target_local, int rows, int local_vocab, int rank, cudaStream_t stream);
```

The contract skeleton is `targets/megatron_5be9626/vocab_parallel_cross_entropy/abi_skeleton.cuh`. It is explicitly not a candidate implementation.

## 4. CUDA 11.8 compatibility

Toolchain profile: `toolchains/v100_sm70_cuda118.json`.

Verified remote facts:

- Tesla V100, `sm_70`, CUDA 11.8
- `cudaStream_t` and `cudaError_t` compile in the target ABI
- `CUDART_INF_F` is forbidden because the real remote nvcc rejected it
- `-3.402823466e+38F` compiled successfully as a finite negative initializer

Probe source: `toolchains/cuda118_negative_initializer_probe.cu`.
Probe result: `toolchains/cuda118_probe_result.txt`.

The Agent prompt now contains a dedicated CUDA 11.8 section and the verified alternative. This is a toolchain fact, not speculative advice.

## 5. Preflight validator

`_preflight_vocab_candidate.py` checks before remote nvcc:

- exact contract marker and hash
- every required ABI symbol
- `extern "C"` visibility pattern
- presence of CUDA stream type
- forbidden CUDA 11.8 tokens

It returns `PASS_PREFLIGHT`, `REJECT_CONTRACT`, or `REJECT_TOOLCHAIN`. It does not claim to replace nvcc.

Observed regression checks:

- Episode 1: `REJECT_TOOLCHAIN`, explicit `forbidden CUDA 11.8 token: CUDART_INF_F`
- Episode 4: `REJECT_CONTRACT`, explicit `missing symbol: local_max_fp16_stream`
- Episode 6: `PASS_PREFLIGHT`

## 6. New Agent prompt

The rebuilt prompt includes:

- exact commit and source path
- TP=2 rank-local semantics
- stage graph and collective invariant
- exact prototypes
- contract hash
- V100/sm_70/CUDA 11.8 profile
- previous compile and ABI failures
- forbidden semantic changes
- no backward implementation

The prompt says that omission of any mandatory symbol invalidates the candidate even if another stage is correct.

## 7. New real episodes

| Episode | Preflight | Compile | TP=1 | TP=2 | Result |
|---:|---|---|---|---|---|
| 6 | PASS | PASS | FAIL correctness | FAIL correctness | `REJECT_CORRECTNESS` |
| 7 | PASS | PASS | not reached | FAIL correctness | `REJECT_CORRECTNESS` |
| 8 | PASS | PASS | not reached | FAIL correctness | `REJECT_CORRECTNESS` |
| 9 | PASS | PASS | not reached | FAIL correctness | `REJECT_CORRECTNESS` |
| 10 | PASS | PASS | not reached | FAIL correctness | `REJECT_CORRECTNESS` |

Episode 6 was explicitly run at TP=1 and failed correctness. Episodes 6–10 were also run against real two-process TP=2; both ranks launched the candidate and reported the same failure. No candidate was benchmarked.

The concrete Episode 6 diagnostic was loss max absolute error `4.42879` at TP=1. TP=2 errors were approximately `4.39346`, `3.74912`, `3.74912`, `4.23096`, and `3.74912` for Episodes 6–10; softmax max absolute error was approximately `0.04350`. The likely semantic issue was fed back: predicted target contribution must use the shifted logit and exp values must use `(logit-global_max)`, with row-shaped target mask semantics. This remains candidate evidence, not a manual repair.

## 8. Collective preservation

Candidates 6–10 were evaluated with the real caller harness: one real MAX all-reduce and two real SUM all-reduces. They did not reach the loss/softmax correctness gate, so they were not eligible for performance or collective-count promotion. No collective was removed, reordered, or emulated.

## 9. Diagnostics and failure containment

Failure classes are now distinct:

```text
REJECT_CONTRACT
REJECT_TOOLCHAIN
REJECT_COMPILE
REJECT_CORRECTNESS
```

Compile failures are contained before distributed launch when preflight catches them. Candidate ABI failures terminate through torchrun without entering candidate collectives. Candidate correctness failures run on both ranks and exit through torchrun; no indefinite hang was observed. The existing SSH timeout remains bounded.

## 10. Benchmark, NSYS, incumbent, replay

The frozen TP=2 baseline was not changed or rerun as a new era. No new candidate reached correctness, so no candidate benchmark, candidate NSYS/NCCL trace, deterministic winning replay, or incumbent was created. The incumbent remains `NO_INCUMBENT`.

## 11. Generic regression and knowledge

The new linter is target-scoped and does not alter standalone evaluator semantics. The normal-candidate preflight path and target prompt generation were exercised by Episode 6: the candidate passed static validation, compiled on CUDA 11.8, and then failed only the correctness gate. New facts are recorded only under `knowledge/targets/megatron_5be9626/vocab_parallel_cross_entropy/`; standalone and SwiGLU knowledge were not modified.

## 12. Upstream integrity

Final verification:

```text
HEAD = 5be9626709af2722333bf54797c954c09edeada3
working tree = clean
```

No Megatron source was modified.

## 13. Remaining blocker

The Agent now reliably satisfies the static ABI and CUDA toolchain gates, and at least one new candidate reaches compilation. The remaining blocker is source-grounded numerical equivalence of the local predicted-logit/softmax stage. Until a new Agent candidate passes TP=1 correctness, Phase 15-B cannot proceed to TP=2 promotion, benchmark, or NSYS candidate profiling.
