# RMSNorm reduction operator context

## Official contract

- Source: `<LOCAL_USER_HOME>\projects\atrex-bench\data\rms_norm`
- Local copied contract: `<PROJECT_ROOT>\ops\rms_norm`
- Input: `x[token_count, hidden_size]` and `weight[hidden_size]`, both `bfloat16` CUDA tensors.
- Output: `out[token_count, hidden_size]`, `bfloat16`.
- Reference: convert `x` to float32, compute `mean(x*x)` over the last dimension, apply `rsqrt(variance + 1e-6)`, multiply by `x` and float32 weight, then cast to the input dtype.
- Official workload: 56 shapes from `shapes.json`; token counts and hidden sizes vary across the supplied contract.

## Reduction structure

Each token row reduces `hidden_size` values to one variance/rstd value, then applies the normalized scale. A CUDA implementation must preserve the float32 accumulation and bfloat16 output semantics. The reduction may require warp/block cooperation; any shared-memory or multi-pass choice must be justified by the candidate itself.

## Official eager reference baseline

Measured with official `atrex_bench.cli.run_eval`, using the immutable official `reference.py` as the input, `--warmup-iters 20`, `--bench-iters 100`, `--perf-timeout-s 0`, and `--skip-kernel-attribution`.

- Compile: 56/56 passed.
- Correctness: 56/56 passed.
- Performance samples: 56/56 present.
- End-to-end latency range: 0.063629 ms to 8.846410 ms.
- Arithmetic mean across the 56 official shapes: 1.243091 ms.
- Example shape 0: tokens=2500, hidden=1024, 0.140806 ms.
- Example shape 1: tokens=4968, hidden=1024, 0.458412 ms.
- Example shape 2: tokens=10000, hidden=1024, 1.139976 ms.
- Example shape 5: tokens=360, hidden=1152, 0.075445 ms.
- Example shape 11: tokens=15000, hidden=1152, 1.993862 ms.

Full result:
`<PROJECT_ROOT>\campaigns\rms_norm\backward\baseline_eager_timeout0\20260910-174616\rms_norm\eval_result.json`

## Constraints

- Target RTX 5060 Laptop GPU, compute capability 12.0, `sm_120`.
- CUDA Toolkit 13.4, PyTorch 2.14.0+cu130.
- Choose exactly one optimization direction.
- Do not modify `reference.py`, official evaluator files, or this baseline.
- Do not use a PyTorch fallback in the candidate.
