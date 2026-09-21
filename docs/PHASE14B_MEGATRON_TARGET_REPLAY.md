# Phase 14-B Reproduction Guide

This guide reproduces the source-grounded SwiGLU observation for Megatron-LM commit `5be9626709af2722333bf54797c954c09edeada3`.

## Local inspection

From aka-local:

```powershell
.\.venv\Scripts\python.exe -m lab.cli target --target swiglu --action inspect
```

This reads the machine-readable target spec without importing optional Megatron runtime dependencies.

## Source and runtime replay

The adapter is `lab/targets/megatron_swiglu.py`. It imports `megatron.core.transformer.mlp.MLP` and executes the exact `MLP.forward` activation branch. The initial branch uses TP=1, `bias_activation_fusion=false`, `use_te_activation_func=false`, and no TE/Apex dependency. The independent replay computes the same documented chunk/SiLU/multiply semantics and then runs the adapter FC2.

The V100 source copy is isolated at `~/aka_targets/megatron-lm-5be9626/` and was verified with `git rev-parse HEAD`. The remote dependency probe uses `<REMOTE_HOME>/venvs/lerobot-act/bin/python`.

## V100 execution

The remote harness is `targets/megatron_5be9626/swiglu/run_remote_swiglu.py`. It records the deterministic fixture metadata, executes the real Megatron class, and prints compact JSON. It does not install packages or modify the global Python environment.

## Profile

The REAL NSYS command used was:

```text
<REMOTE_HOME>/tools/nsys-2026.2.1/target-linux-x64/nsys profile --trace=cuda,nvtx,osrt --sample=none -o /tmp/aka_phase14b_swiglu_nsys <REMOTE_HOME>/venvs/lerobot-act/bin/python /tmp/aka_phase14b_swiglu.py
```

The report was summarized with `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum`.
