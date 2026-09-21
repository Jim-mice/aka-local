# Agent Notes

- Scope: CUDA RMSNorm candidate only.
- `candidate.py` exposes `rmsnorm`, `rms_norm`, `forward`, and `RMSNorm`.
- The implementation performs RMS reduction without mean-centering, accumulates low-precision inputs in FP32, and applies the optional scale vector.
- `hypothesis.json` records the optimization rationale and known integration risks.
- Do not run benchmarks or CUDA as part of this artifact-only task.
