# AKA Local RTX 5060 Status

## Current machine

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- VRAM: 8151 MiB
- Driver: 610.88
- Compute capability: 12.0 / sm_120
- CUDA Toolkit: 13.4.59
- Nsight Compute: 2026.3.0
- MSVC: 19.50.35728 from `D:\Microsoft C++ Build Tools\VC\Tools\MSVC\14.50.35717`
- PyTorch: 2.14.0+cu130

## Smoke

- `smoke/vector_add.py` reaches nvcc + MSVC + PyTorch extension compilation.
- First compile failed because PATH selected MSVC 19.29 and CUDA 13.4 required `/Zc:preprocessor`.
- Correct environment order selected MSVC 19.50; final smoke result is pending after the corrected build output.
- Final vector-add smoke: compile/run passed, `max_abs_error=0.0`, CUDA Event latency about `19.97 us`.

## Operator

- Local copy: `ops/swiglu_forward_v2b`
- Semantics: `out = SiLU(gate) * up`, FP16, M=256/1024/4096, D=4096.
- Recompiled target: `sm_120`; extension name is `aka_local_swiglu_v2b_sm120`.
- Local V2b correctness passed for all three shapes with `max_abs=0.0078125`; the observed FP16 `max_rel` was at most `0.05884`.
- Same-process ABBA benchmark (100 timed repetitions after 20 warmups) passed:
  - M=256: reference 19.60 us, candidate 12.39 us, 1.582x
  - M=1024: reference 33.62 us, candidate 22.33 us, 1.506x
  - M=4096: reference 485.42 us, candidate 290.12 us, 1.673x

## Static and profiler evidence

- `ptxas` compile for `sm_120`: 30 registers/thread, 0 spill stores, 0 spill loads, 0-byte stack frame, 0 barriers.
- Nsight Compute counters are available on the local Windows GPU.
- Report: `profiles/swiglu_kernel_basic.ncu-repz`.
- The filtered V2b kernel was profiled successfully: 256 threads/block, grid 4096, 30 registers/thread, 0 dynamic/static shared memory, theoretical occupancy 100%, achieved occupancy 70.74%, duration about 24.93 us for the profiled launch, memory/DRAM throughput 45.82%, compute throughput 50.92%.

## Remote status

- Crater/V100 is no longer a dependency of the local workflow.

## Current conclusion

- The local RTX5060 CUDA development/evaluation path is operational without the remote gateway.
- V2b half2 fused SwiGLU is locally faster than the PyTorch reference on all requested shapes in the controlled ABBA run.
- NCU is available locally, so the earlier V100 `ERR_NVGPUCTRPERM` restriction does not apply to this workstation.
- Atrex native Windows integration and a new Codex candidate smoke remain separate follow-up work; no new optimization iteration has been started here.

## Codex quota policy

- Primary Agent: Codex CLI
- Primary model: `gpt-5.6-luna`
- Reasoning effort: `low`
- Independent reviewers: disabled for the next campaign
- Max iterations: `1`
- Quota policy: do not auto-escalate to `gpt-5.6-sol` or `gpt-5.6-terra`.
- The local workflow entrypoint sets `ATREX_CODEX_SESSION_SETTINGS` to the Luna/low configuration.
- The explicit Luna/low smoke was attempted on 2026-09-10, but the request timed out and was stopped; no fallback model was used.

## Local Atrex execution episode 1

- Gateway compatibility checkout: `atrex-kernel-agent-win`.
- Gateway health: passed at `http://127.0.0.1:18001/healthz`.
- Sandbox smoke: passed through the Windows gateway to RTX 5060, capability `(12, 0)`.
- Official Atrex-Bench correctness through the gateway: V2b passed `3/3` shapes. Windows used the evaluator's existing `candidate_timeout=0` no-op timeout mode because `signal.SIGALRM` is unavailable on Windows; evaluator semantics and correctness thresholds were unchanged.
- Primary Luna invocation count: `1`.
- Reviewer invocation count: `0`.
- The single Luna request timed out with `stream disconnected` and was stopped during CLI retry `2/5`.
- No candidate was produced and no accept/reject decision was fabricated.
- Episode state: `STOPPED_BY_CODEX_TIMEOUT_BEFORE_CANDIDATE`.

## Luna episode 2

- Existing thread resume was attempted first and returned `no rollout found`; it did not start sampling.
- One fresh primary invocation was then made with `gpt-5.6-luna` and `low` reasoning.
- The fresh request timed out again with `stream disconnected`; it was stopped during retry `2/5`.
- Candidate produced: no. Compile/correctness/ABBA/decision: not applicable.
- Reviewers: 0. Fallback: none. Episode state: `STOPPED_BY_CODEX_TIMEOUT`.

## Frozen project state

The local CUDA/Atrex experiment system is complete and frozen. No further infrastructure work is required before the next Agent attempt.

```ini
LOCAL_INFRA_READY = true
AGENT_BACKEND = codex
MODEL = gpt-5.6-luna
EFFORT = low
REVIEWERS = 0
FALLBACK = none
INCUMBENT = V2b
AGENT_LOOP_COMPLETE = false
BLOCKER = CODEX_REQUEST_TIMEOUT
```

Verified and frozen:

- Windows local gateway
- sandbox to gateway to RTX 5060
- official Atrex-Bench evaluator
- CUDA extension compilation
- V2b canonical correctness
- CUDA Event benchmark and ABBA
- NCU hardware counters
- ptxas/static evidence
- persistent experiment state

The only current blocker is Codex/Luna server-side sampling timeout before hypothesis/candidate generation. Do not rerun infrastructure checks, reinstall dependencies, rerun V2b, rerun NCU, access remote GPUs, or start another model.

## Independent Codex app-server backend probe

- Python package installed in the aka-local venv: `openai-codex==0.147.0` (the available PyPI stable package; it is not the same release number as the selected native runtime).
- Explicit runtime binary: `<LOCAL_USER_HOME>\AppData\Roaming\npm\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe`.
- Runtime reported by app-server initialize metadata: `0.153.4` on Windows.
- Launch mode: independent `codex app-server` over stdio, not Desktop IPC and not `codex exec`.
- Initialize: PASS.
- Existing ChatGPT Plus authentication reuse: PASS via non-sensitive account status; no login/logout or credential changes were performed.
- Model metadata read without sampling and includes `gpt-5.6-luna` with supported `low` reasoning.
- Model sampling in this probe: `0`.
- Probe: `agent_backends/appserver_probe.py`.
- Adapter skeleton: `agent_backends/codex_appserver_agent.py`; it does not run a turn automatically and keeps mechanical evaluation outside the backend.
- Atrex official evaluator files were not modified. No Atrex integration or V3 generation was started.
- Current status: `APP_SERVER_BACKEND_READY_FOR_NEXT_SINGLE_INVOCATION`.

## Mechanical V3 episode 3

- Orchestrator: `run_episode.py`.
- Model calls for mechanical phase: `0`.
- Pipeline: candidate manifest -> official Atrex-Bench compile/correctness -> same-process V2b/V3 A/B/B/A -> strict deterministic decision -> JSON persistence.
- First attempt was preserved as `campaigns/luna/episode_3/mechanical/` and stopped at `REFERENCE_INVALID` because the local experimental reference copy lacked canonical `metadata.json`/`roofline.json`.
- Canonical reference was corrected to `<LOCAL_USER_HOME>\projects\atrex-bench\local_ops\swiglu_forward_v2b`; the evaluator was not modified.
- Retry 1 was preserved at `mechanical_retry1/` and stopped at `COMPILE_FAILED` because its subprocess environment did not expose the already-installed venv Ninja executable.
- The runner now explicitly prepends the aka-local venv Scripts directory and CUDA bin to child PATH; no package was installed during this repair.
- Retry 2 completed on RTX 5060: compile passed for 3/3 shapes, official correctness passed for 3/3 shapes, and ABBA completed with 20 warmups and 100 repeats.
- ABBA result: M=256 speedup `1.025587x`, M=1024 `0.993529x`, M=4096 `0.999810x`, arithmetic-mean speedup `1.006309x`.
- Mechanical policy: reused strict-positive-improvement rule (`min_improvement_pct=0.0`, candidate score must be strictly greater); metric is arithmetic-mean candidate-over-incumbent ABBA speedup.
- Decision: `ACCEPTED` by the persisted result at `campaigns/luna/episode_3/mechanical_retry2/decision.json`.
- The incumbent V2b source was not modified or promoted over.

## Episode 3 robustness verification

- No model calls, agents, reviewers, fallback models, NCU, or source changes were made.
- GPU precheck: RTX 5060 was idle at 0 MiB compute memory; only the Desktop process was listed as a graphics process.
- Frozen V2b and V3 were measured in one process with five independent A/B/B/A batches per shape, 20 warmups and 100 timed repetitions per leg.
- M=256: mean `0.963080x`, median `1.005456x`, std `0.075234`, candidate wins `3/5`.
- M=1024: mean `0.996981x`, median `0.994486x`, std `0.007767`, candidate wins `1/5`.
- M=4096: mean `0.999557x`, median `0.999376x`, std `0.000523`, candidate wins `1/5`.
- Aggregate arithmetic mean speedup: `0.986540x`.
- Aggregate geometric mean speedup: `0.986398x`.
- Aggregate total-time ratio: `0.997679x`.
- Existing strict-positive arithmetic-mean policy: `REJECT` for the repeated robustness aggregate.
- Engineering decision: `KEEP_V2B`; V3 remains preserved as an accepted-by-policy but not promoted candidate.
- Evidence: `campaigns/luna/episode_3/robustness_abba.json`.

## Resume packet for the next available Luna call

Use exactly one primary call with `gpt-5.6-luna` and `model_reasoning_effort=low`; reviewers and fallback remain disabled. Start directly by creating one isolated V3 candidate beyond fusion and half2 in a new episode directory. Do not modify the V2b incumbent, do not add dependencies, and stop model usage as soon as candidate source is written. Then mechanically run candidate compile, canonical correctness, V2b-vs-V3 same-allocation ABBA, and accept/reject. Run targeted NCU only if the result requires diagnosis. If sampling times out again, record `CODEX_REQUEST_TIMEOUT` and stop.

## RMSNorm reduction episode 1

- Official `data/rms_norm` was selected as the reduction-oriented fallback because no official RMSNorm backward or SwiGLU backward contract was present.
- Exactly one Luna Low turn generated the isolated candidate; reviewers and fallback models were disabled.
- Candidate compile and official correctness passed 56/56. Same-process A/B/B/A with 20 warmups and 100 repetitions produced arithmetic mean speedup 7.202740x, geometric mean speedup 6.145711x, and total-time ratio 4.652903x.
- Static evidence: sm_120, 30 registers/thread, 2048 shared bytes/block, zero stack/local bytes, 256 threads/block, one block per row. NCU was not needed after the decisive improvement.
- Decision: `PROMOTE` by mechanical policy; no automatic file replacement was performed. Candidate remains isolated at `campaigns/rms_norm/backward/episode_1/candidate.py`.

## Continuous learning batch

- `rms_norm_v1` is the local incumbent. V1 evidence, lineage, frontier, experience card, lesson, and wiki-candidate inbox were initialized.
- Continuous runner: `continuous_runner.py`; configured for one Luna Low turn per episode, no reviewers, no fallback, and bounded episode/model-turn counts.
- Requested batch: episodes 2-4, maximum 3 primary turns.
- Episode 2 started, but the Luna app-server turn did not return a candidate within the bounded run. It was stopped as `CODEX_REQUEST_TIMEOUT`.
- Episodes 3 and 4 were not started. No CUDA/evaluator/benchmark was run for episode 2, and no candidate or performance result was fabricated.
- Batch state: `STOPPED_BY_CODEX_TIMEOUT`; see `continuous_batch_result.json` and `campaigns/rms_norm/episode_2/decision.json`.

## Resumable episode state semantics

- `EXPLORER_EPISODE = 1`; `LEARNER_EPISODE = 1`; next episode to complete is `2`.
- Episode 2 did not produce `hypothesis.json` or candidate source, so it is not an optimization episode and is not present in GPU experience memory, frontier decisions, or lesson queue.
- Episode 2 is classified as `WAITING_FOR_AGENT`; its last backend attempt is `CODEX_REQUEST_TIMEOUT`.
- Runtime/backend events are separated under `knowledge/runtime_events/`; they are not GPU optimization knowledge.
- Persistent state is `continuous_state.json`. `--resume` keeps episode 2, incumbent, frontier, and model-turn counter instead of advancing or reinitializing them.
- A future complete candidate bypasses the model and enters mechanical evaluation; a hypothesis without candidate remains `CANDIDATE_PENDING`; a timeout with neither leaves `pending_frontier_id` null.
- This maintenance turn made zero model calls and ran no CUDA, evaluator, benchmark, or profiler.

## RMSNorm episode 2 completed

- The Agent consumed episode 1 experience and selected frontier direction `warp_reduce_shared_finalize`.
- Official compile and correctness passed 56/56.
- Five independent same-process A/B/B/A batches across all 56 shapes produced arithmetic mean `1.074215x`, geometric mean `1.071904x`, total-time ratio `1.096522x`, and 50/56 winning shape means.
- Shape 17 (2048x128), previously observed near `0.812x`, won all 5 batches with mean `1.075416x` and std `0.014928`.
- Static resources changed from V1 to V2: registers `30 -> 30`, shared memory `2048 -> 1056 B/block`, stack/local remained zero. NCU was not needed.
- Mechanical decision: `ACCEPT`; local incumbent advanced to `rms_norm_v2`. The official evaluator/reference were unchanged.
- Explorer completed episode: `2`; next episode: `3`. Episode 3 was not started.
