# RMSNorm episode 3 agent instructions

## Objective

Prepare a CUDA RMSNorm forward candidate for the `sm_120` target. This episode records source artifacts only.

## Candidate boundary

- Use one 256-thread block per row.
- Accumulate the sum of squares in FP32 registers.
- Reduce within each warp with `__shfl_down_sync`.
- Store one FP32 partial per warp in shared memory, then finalize in warp 0.
- Preserve FP32 RMS arithmetic, epsilon handling, BF16 weight reads, and BF16 rounded output conversion.
- Keep the two row traversals: reduction first, normalization/output second.

## Prohibited actions

- Do not modify the incumbent, reference implementation, evaluator, or benchmark harness.
- Do not add a fallback implementation or external dependency.
- Do not run CUDA, compilation, benchmarking, evaluation, or profiling for this episode.
- Stop after writing `candidate.py`, `hypothesis.json`, and this `AGENT.md`.

## Provenance

The intended baseline is `C:/Users/38154/projects/aka-local/campaigns/rms_norm/episode_2/candidate.py`. Evaluation remains pending; no runtime or performance claim may be made from these files alone.
