# Agent instructions

## Scope

This directory contains a CUDA RMSNorm optimization candidate. The requested
artifact boundary for this task is exactly:

- `candidate.py`
- `hypothesis.json`
- `AGENT.md`

Do not modify unrelated files.

## Candidate contract

`candidate.py` exposes `rms_norm`, `forward`, `apply`, and an `RMSNorm` module.
The implementation uses PyTorch's fused `torch.nn.functional.rms_norm` backend;
it does not implement a separate handwritten CUDA kernel.

## Safety and validation

- Do not run benchmarks unless explicitly requested.
- Do not run CUDA, compile CUDA extensions, or launch GPU workloads unless
  explicitly requested.
- Do not claim performance, numerical parity, or hardware validation without
  actual measured evidence.
- Keep `hypothesis.json` honest about validation status.

## Editing guidance

Preserve the public function signatures and the normalized trailing-dimension
semantics. If changing epsilon or dtype behavior, update `hypothesis.json` and
record the reason. Keep changes narrowly scoped to this candidate.
