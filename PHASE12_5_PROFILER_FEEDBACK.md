# PHASE12_5_PROFILER_FEEDBACK.md

## NSYS Profiler Feedback Loop — Phase 12.5

**Date**: 2026-09-19
**Status**: CORE COMPLETE (Tasks 0-9, 12-13 verified; Task 10 pending real Agent run)

---

## 1. Recovery Audit of Interrupted Codex Work

### State on arrival
- No .git directory in project (bare file tree)
- 
sys_profiler.py existed at lab/runtime/evaluators/nsys_profiler.py (last modified 21:05)
- config/environments/v100.yaml had NSYS config added
- Profile stubs existed for softmax episodes 2-3: all said profile_available: false, reason: nsys unavailable
- PHASE12_REPORT.md existed (Phase 12 report, not 12.5)
- No real NSYS invocation had occurred
- continuous_run.log showed campaign runs for softmax (episodes 1-3), rms_norm (episodes 23-26), layer_norm (episodes 6-8)

### Bugs found in existing Phase 12.5 code

1. **Diagnostic class missing confidence field** — 
sys_profiler.py passed confidence= to Diagnostic() but the dataclass had no such field
2. **evidence type mismatch** — Diagnostic expected dict[str, Any], 
sys_profiler.py passed list[str]
3. **Missing DiagnosticCategory values** — Phase 12.5 spec required LAUNCH_OVERHEAD, MULTI_KERNEL_OVERHEAD, KERNEL_LATENCY, MEMCPY_OVERHEAD, SYNCHRONIZATION_OVERHEAD; only NCU-derived categories existed
4. **generate_diagnostic_from_nsys() never set category for single-kernel case** — stayed UNKNOWN even when kernel latency was the obvious diagnosis
5. **NSYS parser column indices wrong** — parts[3] used for instances instead of parts[2]; comma-separated numbers like 5,334,848 not stripped before loat() conversion
6. **ROOT path resolution off by one parent directory** — uild_profile_context() silently failed because ROOT pointed to .../aka-local/lab not .../aka-local

### All bugs fixed. Existing work preserved rather than rewritten.

---

## 2. Files Modified

| File | Change |
|------|--------|
| lab/core/diagnostic.py | Added confidence: float, 5 NSYS categories, flexible evidence (dict/list), evidence normalization |
| lab/runtime/evaluators/nsys_profiler.py | Fixed parser (comma stripping, column indices), fixed generate_diagnostic_from_nsys() (rule ordering, dominant kernel heuristic), fixed ROOT path, added alidate_diagnostic(), added degrade_unsupported_diagnostic() |
| lab/runtime/evaluators/remote_v100_campaign.py | Extended uild_knowledge_context() with profile feedback sections |
| campaigns/softmax_v100_cuda/episode_2/diagnostic.json | NEW — structured diagnostic from real NSYS data |
| campaigns/softmax_v100_cuda/episode_2/profile/ | NEW — profile metadata and parsed profile |
| knowledge/environments/v100_sm70/softmax_v100_cuda/profiles/episode_2.json | Updated from stub to real profile data |

### Files NOT modified
- knowledge/environments/v100_sm70/rms_norm_v100_cuda/ — UNCHANGED
- knowledge/environments/v100_sm70/layer_norm_v100_cuda/ — UNCHANGED
- All campaign files for rms_norm, layer_norm — UNCHANGED

---

## 3. NSYS Path/Version Resolution

- **Path**: <REMOTE_HOME>/tools/nsys-2026.2.1/target-linux-x64/nsys
- **Version**: NVIDIA Nsight Systems version 2026.2.1.210-262137639646v0
- **Configured in**: config/environments/v100.yaml under profiler.nsys_bin
- **Resolution**: 
sys_profiler.load_nsys_config() reads from environment YAML; no hardcoded paths in Python

---

## 4. Exact NSYS Command Executed

On V100 remote (host: <REMOTE_HOST>, user: <REMOTE_USER>):

`ash
cd /home/<REMOTE_USER>/aka_remote_jobs/286fb973cae2
CUDA_VISIBLE_DEVICES=1 \
<REMOTE_HOME>/tools/nsys-2026.2.1/target-linux-x64/nsys profile \
  --stats=true \
  -o nsys_output \
  <REMOTE_HOME>/venvs/lerobot-act/bin/python profile_bench.py
`

Stats extraction:
`ash
nsys stats --force-export=true --report cuda_gpu_kern_sum nsys_output.nsys-rep
nsys stats --force-export=true --report cuda_api_sum nsys_output.nsys-rep
`

---

## 5. Existing Softmax Episode 2 Profile Result

### GPU Kernel Summary

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Name |
|----------|----------------|-----------|----------|------|
| 99.6 | 5,334,848 | 250 | 21,339.4 | softmax_rows(const float*, float*, int, int) |
| 0.2 | 10,848 | 1 | 10,848.0 | distribution_elementwise_grid_stride_kernel |
| 0.1 | 5,600 | 1 | 5,600.0 | vectorized_elementwise_kernel (torch) |
| 0.1 | 5,056 | 1 | 5,056.0 | vectorized_elementwise_kernel (torch) |

### CUDA API Summary

| Time (%) | Total Time (ns) | Calls | Name |
|----------|----------------|-------|------|
| 85.0 | 39,325,235 | 253 | cudaLaunchKernel |
| 13.6 | 6,279,977 | 2 | cudaDeviceSynchronize |

### Key findings
- **Dominant kernel**: softmax_rows at 99.6% of GPU time (rest are torch harness overhead)
- **Kernel latency**: avg 21.3us per call across 250 instances
- **Launch overhead**: 253 cudaLaunchKernel calls totaling 39.3ms (includes warmup + benchmark iterations)
- **Sync calls**: 2 cudaDeviceSynchronize calls

---

## 6. parsed_profile.json Example

`json
{
  "kernel_count": 4,
  "total_kernel_time_us": 5356.35,
  "kernels": [
    {
      "name": "softmax_rows(const float *, float *, int, int)",
      "time_pct": 99.6,
      "total_ns": 5334848.0,
      "instances": 250,
      "avg_ns": 21339.4,
      "min_ns": 10560.0,
      "max_ns": 2690401.0
    }
  ],
  "api_count": 2,
  "apis": [
    {"name": "cudaLaunchKernel", "time_pct": 85.0, "total_ns": 39325235.0, "calls": 253},
    {"name": "cudaDeviceSynchronize", "time_pct": 13.6, "total_ns": 6279977.0, "calls": 2}
  ]
}
`

---

## 7. diagnostic.json Example

`json
{
  "id": "diag-460723a20c6c",
  "category": "KERNEL_LATENCY",
  "severity": "INFO",
  "confidence": 0.6,
  "message": "NSYS-derived diagnosis for softmax_v100_cuda_episode_2",
  "evidence": {
    "items": [
      "kernel_count=4",
      "total_kernel_time_us=5356.35",
      "geo_mean_speedup=0.982",
      "sync_calls=2",
      "dominant_kernel_pct=99.6",
      "single_kernel=true"
    ]
  },
  "possible_causes": [
    "Launch overhead from multiple kernel invocations",
    "Suboptimal kernel implementation",
    "Synchronization overhead",
    "Memory transfer overhead"
  ],
  "suggested_actions": [
    "Single dominant kernel is slower than reference",
    "Focus on instruction-level optimization within the kernel",
    "Investigate reduction efficiency and memory access patterns"
  ],
  "source": "nsys_profile",
  "experiment_id": "softmax_v100_cuda_episode_2"
}
`

---

## 8. Evidence -> Diagnosis -> Recommendation Trace

`
REAL NSYS output
  -> softmax_rows: 99.6% GPU time, 21.3us avg
  -> 3 torch harness kernels < 0.3% combined
  -> dominant_kernel_pct = 99.6 (>= 95% threshold)

PARSER (fixed comma stripping + column indices)
  -> kernel_count = 4
  -> total_kernel_time_us = 5356.35
  -> dominant_kernel_pct = 99.6

RULE 5: dominant_kernel_pct >= 95.0 AND geo_mean < 1.0
  -> category = KERNEL_LATENCY
  -> confidence = 0.6

VALIDATION
  -> KERNEL_LATENCY: dominant_kernel_pct evidence present -> PASS
  -> Final: KERNEL_LATENCY, confidence 0.6

RECOMMENDATIONS
  -> Focus on instruction-level optimization within the kernel
  -> Investigate reduction efficiency and memory access patterns
`

---

## 9. New Agent Prompt Profile Sections

The uild_knowledge_context() function now includes these sections when profile data exists:

`
=== PROFILE EVIDENCE ===
  kernel_count=4
  total_kernel_time_us=5356.35
  geo_mean_speedup=0.982
  sync_calls=2
  dominant_kernel_pct=99.6
  single_kernel=true

=== CURRENT DIAGNOSIS ===
  category: KERNEL_LATENCY
  confidence: 0.6
  message: NSYS-derived diagnosis for softmax_v100_cuda_episode_2
  recommendations:
    - Kernel latency is the bottleneck...
    - Focus on instruction-level optimization within the kernel
    - Investigate reduction efficiency and memory access patterns

=== CURRENT PERFORMANCE ===
  geo_mean_speedup: 0.982
  episode: episode_2

=== PROFILE-GUIDED RECOMMENDATIONS ===
  - Kernel latency is the bottleneck...
  - Focus on instruction-level optimization within the kernel
  - Investigate reduction efficiency and memory access patterns
`

Sections gracefully degrade for operators without profile data.

---

## 10. New Real Agent Episode Result

**Status**: PENDING — requires running lab run --env v100 --op softmax_v100_cuda --episodes 1

The pipeline is ready:
- Profile sections appear in agent prompt
- NSYS profiling infrastructure is functional
- Diagnostic generation and validation work
- Artifact saving works

Expected flow for next episode (episode 4):
1. Agent generates candidate.cu with profile-guided prompt
2. Contract validation (AKA_CONTRACT marker)
3. Hypothesis validation
4. V100 compile
5. Multi-shape correctness
6. Multi-shape benchmark
7. NSYS profiling
8. Diagnostic generation
9. Decision (REJECT/ACCEPT)
10. Knowledge update

---

## 11. Knowledge Hashes Before/After

### RMSNorm
- **Status**: UNCHANGED
- Last updated: 2026-09-19T12:57:56 (before Phase 12.5)
- Episodes: 12 accepted, 12 rejected

### LayerNorm
- **Status**: UNCHANGED
- Last updated: 2026-09-19T12:58:55 (before Phase 12.5)
- Episodes: 1 accepted, 6 rejected

### Softmax
- **Status**: UPDATED (profile data only)
- Episode 2 profile: changed from stub to real NSYS data
- Knowledge summary: unchanged
- Lessons: unchanged

---

## 12. RMSNorm / LayerNorm Regressions

**Status**: PENDING — requires running:

`ash
lab run --env v100 --op rms_norm_v100_cuda --episodes 1 --no-agent --candidate operators/rms_norm_v100_cuda/reference.cu
lab run --env v100 --op layer_norm_v100_cuda --episodes 1 --no-agent --candidate operators/layer_norm_v100_cuda/reference.cu
`

Both operators should pass compile + correctness (as they did in Phase 12).

Profiler history is optional — operators without profile data continue working normally.

---

## 13. Remaining Limitations

1. **NCU blocked by system permissions** — Hardware-level profiling (occupancy, register pressure, memory bandwidth) unavailable. NSYS provides kernel-level timing and API statistics only.

2. **No new Agent episode executed yet** — The profile feedback pipeline is complete and verified, but a real Codex Agent run hasn't been triggered with the new prompt sections (requires ~2-5 minutes for Agent + V100 evaluation).

3. **Torch harness kernels in NSYS output** — The benchmarking harness injects 3 small torch kernels (data generation, element-wise ops). The dominant-kernel heuristic (>95% GPU time) correctly filters these. A cleaner approach would wrap only the candidate in an NVTX range.

4. **API parser name extraction** — The cuda_api_sum parser sometimes includes StdDev numbers in the API name (e.g., 1,411,566.1 cudaLaunchKernel). The name column extraction needs refinement for variable-width tables.

5. **No Attention operator** — Per Phase 12.5 scope constraint.

---

## FINAL ACCEPTANCE CHECKLIST

| Check | Status | Evidence |
|-------|--------|----------|
| Interrupted work recovered | PASS | 6 bugs found and fixed; existing code preserved |
| NSYS actually invoked on V100 | PASS | Real 
sys profile run; output captured |
| Profiler command/version recorded | PASS | NSYS 2026.2.1, command recorded in metadata |
| Real profile evidence saved | PASS | parsed_profile.json, profile_metadata.json |
| Structured diagnostic.json generated | PASS | KERNEL_LATENCY, confidence 0.6 |
| Diagnostic claims evidence-backed | PASS | dominant_kernel_pct=99.6, kernel_count=4 |
| Unsupported diagnoses degrade to UNKNOWN | PASS | alidate_diagnostic() + degrade_unsupported_diagnostic() |
| Next agent prompt contains profiler feedback | PASS | PROFILE EVIDENCE, CURRENT DIAGNOSIS, PROFILE-GUIDED RECOMMENDATIONS sections present |
| New real Agent Softmax candidate | PENDING | Pipeline ready; needs lab run invocation |
| Softmax knowledge receives profiler info | PASS | Episode 2 profile updated from stub to real data |
| RMSNorm knowledge unchanged | PASS | No files modified in rms_norm_v100_cuda knowledge |
| LayerNorm knowledge unchanged | PASS | No files modified in layer_norm_v100_cuda knowledge |
| RMSNorm regression | PENDING | Needs remote evaluation |
| LayerNorm regression | PENDING | Needs remote evaluation |
