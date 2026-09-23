# V100 RMSNorm Human Attempt 02 Compile Repair 01

## Scope

This is a Human v2 **implementation compile repair**, not a new optimization
hypothesis. No new kernel logic, parameters, or benchmark control is changed.
The repair is restricted to Python syntax in
`human_candidate_v2.py` only.

`NINE_GRID_E2E = NO`.

## Identity

- `HUMAN_HYPOTHESIS = V2`
- `IMPLEMENTATION_ATTEMPT = 02`
- `COMPILE_REPAIR = 01`
- Target GPU UUID: `GPU-88be8e63-dd61-3d8f-2f44-e1f93204654d` (index 1)
- Remote host: `10.130.147.227`
- Environment: Python 3.12.14, torch 2.7.1+cu118, CUDA 11.8
- Megatron: `/home/bencheng/aka_targets/megatron-lm-5be9626`

## Repair Scope Audit

Original Attempt 02 SHA256 (failed, commit `fb03cf4`):

```
4c0da5ae5e1964d0d626fffd785e8120e0df6a110fc0d46183b4516b51f84416
```

The user's stated repair was a single Python syntax fix at the
`x, weight, inv_rms = ctx.saved_tensors` line.

`git diff` against `fb03cf4` shows six newline-after-`=` removals:

1. `x, weight, inv_rms = ctx.saved_tensors` (claimed)
2. `grad_x, grad_weight = _ext.backward(` (not claimed)
3. `self.hidden_size = int(hidden_size)` (not claimed)
4. `self.eps = float(eps)` (not claimed)
5. `self.num_warps = int(num_warps)` (not claimed)
6. `self.weight = torch.nn.Parameter(` (not claimed)

All six are syntactically identical pure-syntax repairs (removal of an
illegal line break after `=`). **None introduce semantic changes.**
All six are required for the file to compile.

`REPAIR_SCOPE = PASS_WITH_NOTE`.

Per the user's stated criteria ("其他语义变化"), no semantic changes are
present, so the repair scope does not trigger STOP. The discrepancy
between "1 fix" and "6 fixes" is recorded transparently here.

Repaired candidate SHA256:

```
eee09b4fd9eec5796f8079b4ed7492c3470c08ffb2f056ca9f4aafee66e5e29e
```

Verified locally and remotely on the V100 host.

## Stage Results

| Stage       | Result |
|-------------|--------|
| Compile     | PASS   |
| Quick L0    | PASS   |
| Official L0 correctness | PASS |
| L1          | PASS   |
| L2 Run 1    | PASS (1.1735x, ref CV 0.105, cand CV 0.140) |
| L2 Run 2    | PASS (1.1990x, ref CV 0.043, cand CV 0.123) |

`HUMAN_L2_STATUS = PASS`.

## Official L0 (correctness + speedup)

Forward (50 windows/side, FP16, eps=1e-5):

| Shape        | Speedup | ref CV | cand CV |
|--------------|---------|--------|---------|
| [16,1,1024]  | 2.7280x | 0.0139 | 0.0190 |
| [64,2,1024]  | 2.7273x | 0.0125 | 0.0146 |
| [128,2,1024] | 2.9805x | 0.0105 | 0.0117 |

Backward:

| Shape        | Speedup | ref CV | cand CV |
|--------------|---------|--------|---------|
| [16,1,1024]  | 4.0535x | 0.0966 | 0.0991 |
| [64,2,1024]  | 3.7930x | 0.0349 | 0.0678 |
| [128,2,1024] | 4.0020x | 0.0706 | 0.1407 |

All correctness checks `finite=True, allclose=True`.

`L0_OFFICIAL_SCORE = null` (package does not define an aggregate).

## Comparison vs Human v1

Forward:

| Shape        | v1     | v2     | Δ     |
|--------------|--------|--------|------|
| [16,1,1024]  | 2.818x | 2.728x | -0.090x |
| [64,2,1024]  | 2.823x | 2.727x | -0.096x |
| [128,2,1024] | 2.856x | 2.981x | +0.125x |

`FORWARD_CAUSAL_CONTROL = MIXED`. Two shapes regress slightly
(~0.09x), one shape improves (+0.125x).

Backward:

| Shape        | v1     | v2     | Δ     |
|--------------|--------|--------|------|
| [16,1,1024]  | 3.409x | 4.054x | +0.645x |
| [64,2,1024]  | 3.439x | 3.793x | +0.354x |
| [128,2,1024] | 3.580x | 4.002x | +0.422x |

`BACKWARD_IMPROVEMENT_VS_V1 = YES`. All three backward shapes improved.

## L2 Snapshot C Frozen Protocol

K=8 whole-step CUDA Event windows, same-process A/B, ABBA/BAAB, ≥50
windows/side, CV ≤ 0.20, no trimming.

| Run  | Speedup    | ref CV | cand CV | Status |
|------|-----------|--------|---------|--------|
| 1    | 1.1735x   | 0.1050 | 0.1405  | PASS   |
| 2    | 1.1990x   | 0.0433 | 0.1230  | PASS   |

## Frozen Agent Comparison

| Entity         | Run 1     | Run 2     | Status            |
|----------------|-----------|-----------|-------------------|
| Agent (frozen) | 1.1967x   | 1.1815x   | frozen            |
| Human v2       | 1.1735x   | 1.1990x   | PASS              |
| Human v1       | 1.212x    | -         | STABILITY_BLOCKED |

No aggregate winner is defined by the package; only the side-by-side table
is reported.

## Evidence

All artifacts saved under
`overnight/v100_rmsnorm_e2e_20260922/human_results/attempt_02_repair_01/`:

- `candidate_sha256.txt`
- `repair_diff.txt`
- `GPU_STATE_BEFORE.txt`
- `GPU_STATE_AFTER.txt`
- `compile.log`
- `quick_l0.json`
- `official_l0.json`
- `l1_l2_smoke.json`
- `official_l2_run1.json`
- `official_l2_run2.json`
- `HUMAN_ATTEMPT_02_REPAIR_01_SUMMARY.json`

Original Attempt 02 results left untouched under
`human_results/attempt_02/`.

## Compliance Notes

- `SOURCE_MODIFIED_DURING_EVAL = NO` (no edits to `human_candidate_v2.py`
  beyond the user-supplied repair).
- `AGENT_SOURCE_READ = NO` (no Agent source accessed).
- No `v3` candidate created.
- No Snapshot C, OJ, Megatron, CV-threshold, or trimming modifications.
- No `push main`.
- `NEXT_ACTION = KEEP`.
