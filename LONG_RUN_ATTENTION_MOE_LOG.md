# Overnight Attention/MoE Long-Run Log

## Start

- Phase 17-B historical state: `BUILD_BLOCKED`.
- A1 remains immutable and rejected because CUDA 11.8 reports `CUDART_INF_F` undefined.
- Active attention contracts and reference baseline remain frozen.
- Next action: generic preflight hardening, compatible sentinel compile probe, then authorized A2.

## Phase 17-B.1 toolchain recovery

- Generic `validate_candidate_contract_marker` now rejects `CUDART_INF_F` before nvcc.
- CUDA 11.8 / sm_70 probe passed with finite negative literal and `-INFINITY`.
- `v100-srv` alias was not resolvable in this session; bounded fallback through the existing project adapter completed without exposing credentials.
- A2 is authorized as toolchain-feedback-guided; A1 remains immutable.
