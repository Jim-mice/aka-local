# Phase 14-D Final Acceptance

## 1. Provenance and policy

Episode 2 candidate hash: `21e7995911f9af2d`.
Stream-aware integration adapter hash: `281fa3160db58a73`.
Optimization contract: `2b05cc321bed0956`.
Integration contract: `00b8d9b26b2dc005`.
Megatron commit: `5be9626709af2722333bf54797c954c09edeada3`.

The frozen policy is controlled error, not fallback, for unsupported dtype,
layout, device, or shape. The wrapper raises `REJECT_INTEGRATION` before any
CUDA launch. There is no runtime metadata activation gate for candidate,
contract, or adapter hashes; contract-mismatch testing is therefore recorded as
not applicable rather than simulated by mutating authoritative artifacts.

## 2. Negative tests

Evidence: `targets/megatron_5be9626/swiglu/negative_tests.json`.
All tests ran on the V100 using the actual wrapper and compiled integration
adapter.

| Fixture | Result | Evidence |
|---|---|---|
| FP32 input | PASS | `REJECT_INTEGRATION: expected CUDA FP16 intermediate` |
| Non-contiguous CUDA input | PASS | `REJECT_INTEGRATION: intermediate must be contiguous` |
| Unsupported rows (`[3,8192]`) | PASS | `REJECT_INTEGRATION: unsupported official shape` |
| CPU FP16 input | PASS | `REJECT_INTEGRATION: expected CUDA FP16 intermediate` |
| Contract mismatch | N/A | no runtime metadata gate exists |

Each rejected case recorded `candidate_launched: false`. Validation precedes
allocation and dispatch, so invalid tensors are not reinterpreted as FP16 CUDA
storage and no incompatible kernel was launched. Because the documented policy
is error rather than fallback, fallback-output comparison is not applicable.

## 3. Positive regression

After the negative tests, the supported `[16,1,1024]`, `[64,2,1024]`, and
`[128,2,1024]` paths were rerun. Forward correctness passed with maximum
absolute errors `6.10e-5`, `1.22e-4`, and `1.22e-4`. The optimized
`launch_swiglu_stream` path remained active. Gradient regression passed:
input-gradient max absolute error `1.4114e-4`; FC1 and FC2 weight-gradient
max absolute error `0.00390625`.

## 4. Performance separation

No performance history was overwritten:

- activation boundary: approximately `2.83x`
- complete MLP forward: approximately `1.023x`
- complete MLP forward+backward: `0.918x`

The last figure is not an optimized training result; backward remains trusted
PyTorch analytical code.

## 5. Integrity

The authoritative checkout was rechecked at the end:

`HEAD = 5be9626709af2722333bf54797c954c09edeada3`

`git status --short` was empty. No Megatron source file was modified.

## 6. Acceptance matrix

| Requirement | Verdict |
|---|---|
| Wrong dtype safe handling | PASS |
| Non-contiguous safe handling | PASS |
| Unsupported dimension safe handling | PASS |
| Device validation | PASS |
| Contract mismatch | N/A; no runtime gate exists |
| Invalid inputs never launch candidate | PASS |
| Valid input uses optimized candidate | PASS |
| Valid forward correctness | PASS |
| Valid gradients | PASS |
| Megatron checkout untouched | PASS |

## Final verdict

**PHASE 14-D FULL PASS**, with contract-mismatch validation explicitly not
applicable because the current sidecar has no runtime metadata activation gate.
No new candidate generation, kernel optimization, boundary expansion, or
Megatron upstream modification occurred.
