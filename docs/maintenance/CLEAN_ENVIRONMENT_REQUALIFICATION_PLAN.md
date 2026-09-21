# Clean-Environment Requalification Plan

This is a plan, not authorization to run a campaign. Each permitted retry is
one pre-authorized, complete, independent campaign with fixed physical mapping,
identical contract/scope/policy, read-only environment capture before and after
each invocation, and no merging with historical samples. It must not be
repeated until CV passes.

| Target | Immutable candidate | Required frozen scope | Historical evidence | New-pass condition |
|---|---|---|---|---|
| CE backward | B1 | rank-local backward, 0 collectives, TP=2 policy | raw B1/requalification rows remain historical | complete paired campaign under existing CV ≤0.20 policy |
| Torch RMSNorm | R1/R2 | WrappedTorchNorm forward FP16 out-of-place scope | raw R1/R2 non-promotable rows remain historical | complete paired campaign under existing policy |
| MoE SequentialMLP | M4 `1fb2921a0e0706d2e71bc42073036ca3b048661edaa7c1e76ae22f2e61e9e549` | E=4,H=64,I=128 distributions [4,4,4,4], [1,3,5,7], [0,0,8,8] | M4 raw rows, including 4.99 ms reference event, remain historical | one full frozen-policy M4 campaign; all reference/candidate distribution CVs ≤0.20 |

A cleaner window means stable observed hardware/process state before start; it
does not authorize discarding later slow observations. Any changed mapping,
fixture, contract, candidate hash, or scope creates a separate era rather than
a comparison with old data.
