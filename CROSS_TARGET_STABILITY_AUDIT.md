# Cross-Target Stability Audit

| Target | Side/config affected | Evidence | Verdict |
|---|---|---|---|
| CE backward | paired reference rank 0 on two official configs | CV 0.209 and 0.223, valid 717.856/1010.688 us observations; host already had unrelated CUDA process and nonuniform GPU state | `INSUFFICIENT_EVIDENCE` for a physical-GPU cause; reference instability is confirmed. |
| Torch RMSNorm | candidate R1/R2 | candidate CV gate failed despite correctness | `INSUFFICIENT_EVIDENCE`; prior reports do not establish a shared GPU or host cause. |
| MoE M4 | reference, [0,0,8,8] | CV 0.3733; 4.9899 ms event at invocation 2/block 2/sample 0, with candidate neighbors 1.1769/1.1711 ms and preceding reference 1.0963 ms | `REFERENCE_ONLY_LOCAL_OR_SYSTEM_DELAY_UNKNOWN`. |

No common physical-GPU correlation can be asserted: historical telemetry is
incomplete and target eras differ. No valid observations were deleted. The
generic readiness capture exists to improve attribution before future campaigns.
