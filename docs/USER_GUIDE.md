# AKA-Local User Guide

## Beginner Workflow

### 1. Verify your environment

```bash
python -m lab.cli doctor --env v100
```

This checks SSH connectivity, GPU availability, CUDA compiler, evaluator,
knowledge integrity, and configuration. All 6 checks must pass.

If the evaluator is missing, run `scripts/setup_v100.sh` on the V100 server.

### 2. See what operators are available

```bash
python -m lab.cli list-ops
```

Shows V100 CUDA operators, registry operators, and RTX5060 operators
with their interfaces and entry points.

### 3. Evaluate a candidate kernel

```bash
python -m lab.cli evaluate --candidate path/to/candidate.cu --env v100
```

Compiles the candidate on V100, runs correctness checks on all shapes,
and reports per-shape latency and speedup vs PyTorch reference.

Example with the built-in reference:
```bash
python -m lab.cli evaluate --candidate operators/rms_norm_v100_cuda/reference.cu --env v100
```

### 4. Run optimization

With Codex agent (autonomous):
```bash
python -m lab.cli run --env v100 --op rms_norm_v100_cuda --episodes 1
```

Without Codex agent (manual candidate):
```bash
python -m lab.cli run --env v100 --op rms_norm_v100_cuda --no-agent
```

Resume an interrupted campaign:
```bash
python -m lab.cli run --env v100 --op rms_norm_v100_cuda --resume-run
```

### 5. View the report

```bash
python -m lab.cli report --env v100 --op rms_norm_v100_cuda
```

Shows current incumbent, best strategies, failures, and recent experiment history.

---

## Advanced Workflow

### Creating a custom candidate.cu

Create a standalone .cu file with the required interface:

```c
extern "C" void launch_kernel(
    float* x,       // input [batch, hidden]
    float* weight,  // weight [hidden]
    float* y,       // output [batch, hidden]
    int batch,
    int hidden,
    float eps
);
```

Then evaluate it:
```bash
python -m lab.cli evaluate --candidate my_kernel.cu --env v100
```

Or run it as part of a campaign:
```bash
cp my_kernel.cu campaigns/rms_norm_v100_cuda/episode_N/candidate.cu
python -m lab.cli run --env v100 --op rms_norm_v100_cuda --no-agent
```

### Replaying an episode

Verify an episode's results are reproducible:
```bash
python -m lab.cli replay --env v100 --op rms_norm_v100_cuda --ep 16
```

This re-runs the evaluation and compares the score. Deviation > 2% is reported.

### Inspecting knowledge

```bash
python -m lab.cli knowledge --operator rms_norm_v100_cuda
```

Shows accumulated knowledge: successful strategies, failures, recommendations.

### Recovering interrupted runs

```bash
python -m lab.cli recover --list
python -m lab.cli recover --inspect RUN_ID
python -m lab.cli recover --resume RUN_ID
```

---

## Command Reference

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `doctor` | Environment health check | `--env v100` |
| `list-ops` | List all operators | |
| `evaluate` | Evaluate a candidate.cu | `--candidate PATH --env v100` |
| `run` | Run optimization campaign | `--env --op --episodes --no-agent --resume-run` |
| `replay` | Replay an episode | `--env --op --ep N` |
| `report` | Generate report | `--env --op` |
| `status` | Show project status | |
| `recover` | Recover interrupted runs | `--list --inspect ID --resume ID` |

---

## File Organization

```
aka-local/
  lab/cli.py              # Main CLI entry point
  scripts/setup_v100.sh   # V100 evaluator setup
  config/environments/    # V100 config (edit this!)
  operators/              # V100 CUDA operators
  campaigns/              # Experiment data
  knowledge/environments/ # Per-environment learned knowledge
  docs/                   # Documentation
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| SSH fails | Check v100.yaml host/user, check .v100_secret password |
| Evaluator missing | Run scripts/setup_v100.sh on V100 |
| nvcc not found | Install CUDA 11.8 on V100 |
| Compile fails | Check candidate.cu syntax, verify CUDA arch=sm_70 |
| Correctness fails | Check kernel bounds, verify all shapes produce output |
| No operators listed | Check operators/ directory has metadata.json |
