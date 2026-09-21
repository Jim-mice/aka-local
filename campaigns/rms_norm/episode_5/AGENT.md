# Agent instructions

## Scope

This directory contains one CUDA RMSNorm optimization candidate. The artifact
boundary for this task is exactly:

- `candidate.py`
- `hypothesis.json`
- `AGENT.md`

Do not modify unrelated files.

## Candidate contract

`candidate.py` exposes `rms_norm`, `forward`, `apply`, and an `RMSNorm` module.
The implementation delegates to `torch.nn.functional.rms_norm`, preserving
trailing-dimension semantics and avoiding a handwritten extension or extra
normalization intermediates.

## Safety and validation

- Do not run benchmarks unless explicitly requested.
- Do not run CUDA, compile CUDA extensions, or launch GPU workloads unless explicitly requested.
- Do not claim performance, numerical parity, or hardware validation without actual measured evidence.
- Keep `hypothesis.json` honest about validation status.

## Editing guidance

Preserve the public function signatures, epsilon default, affine weight shape,
and normalized trailing-dimension semantics. Keep changes narrowly scoped to
this candidate and update `hypothesis.json` if the implementation strategy or
validation status changes.
