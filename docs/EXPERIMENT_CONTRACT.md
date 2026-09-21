# Experiment Contract

## Candidate Interface

Every candidate must implement exactly:

`c
extern "C" void launch_kernel(
    float* x,       // input tensor [batch, hidden]
    float* weight,  // weight tensor [hidden]
    float* y,       // output tensor [batch, hidden]
    int batch,      // batch dimension
    int hidden,     // hidden dimension
    float eps       // epsilon for numerical stability
);
`

- **File**: candidate.cu (standalone .cu file)
- **No external dependencies**: No PyTorch, no Python
- **Compiler**: nvcc with flags: -gencode arch=compute_70,code=sm_70 -O2

## Correctness

- **Tolerance**: max_error <= 0.001 (1e-3)
- **Validation**: Output compared against reference PyTorch RMSNorm
- **All shapes must pass**: Every shape in the evaluation config must pass correctness
- **Single-shape passing does NOT qualify**: Episode 14 is the canonical example

## Benchmark

- **Warmup**: 50 iterations (discarded)
- **Measurement**: 200 timed iterations
- **Metric**: Geometric mean speedup over reference PyTorch RMSNorm
- **Score type**: geometric_mean_speedup

## Score

score = geometric_mean(speedup_shape_1, speedup_shape_2, ..., speedup_shape_N)

Where speedup_shape_i = reference_latency_i / candidate_latency_i

## Shapes

Default evaluation shapes (from config/environments/v100_sm70/evaluation.json):

| Batch | Hidden |
|-------|--------|
| 1     | 4096   |
| 4     | 4096   |
| 8     | 4096   |
| 32    | 4096   |

## Promotion Rule

- Candidate score > incumbent score (strict greater-than)
- All shapes pass correctness
- Compile succeeds
- Margin: 0% (any improvement is accepted)

## Replay Rule

Replay must reproduce the original score within 2% deviation.

Replay uses:
- Same candidate.cu (verified by SHA256 hash)
- Same shapes (verified by contract hash)
- Same baseline reference (verified by SHA256 hash)
- Same evaluator (verified by evaluator hash)
- Same evaluation config (warmup, iterations, compiler flags)

If replay deviates > 2%:
1. Check candidate hash matches manifest
2. Check evaluator has not changed
3. Check baseline has not changed
4. Record evidence in replay_result.json

## Episode Manifest

Every episode records:
`json
{
  "episode": N,
  "operator": "rms_norm_v100_cuda",
  "contract_hash": "<SHA256[:16] of shapes>",
  "candidate_hash": "<SHA256[:16] of candidate.cu>",
  "baseline_hash": "<SHA256[:16] of reference.cu>",
  "evaluation": {
    "shapes": ["1,4096", "4,4096", "8,4096", "32,4096"],
    "score_type": "geometric_mean_speedup"
  }
}
`

## Legacy Warning

Episodes before Phase 8-C used single-shape evaluation.
These are marked with legacy=true in their metadata.
Single-shape scores cannot be compared to multi-shape scores.

## Environment Isolation

Knowledge is strictly isolated by environment:
- knowledge/environments/v100_sm70/ - V100-specific
- knowledge/environments/rtx5060_sm120/ - RTX5060-specific

No cross-contamination is permitted.
