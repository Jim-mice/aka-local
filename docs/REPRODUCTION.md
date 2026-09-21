# AKA-Local Reproduction Guide

## Requirements

### Hardware
- **V100 GPU**: NVIDIA Tesla V100-PCIE-16GB (or compatible) accessible via SSH
- CUDA 11.8+ on the V100 server
- nvcc compiler on V100 server

### Local Machine
- **OS**: Windows 10+ or Linux (tested on Windows PowerShell)
- **Python**: 3.11+
- **SSH client**: OpenSSH (built into Windows 10+/Linux)
- **Network**: SSH access to V100 server (default: port 22)

### Python Packages
```
paramiko>=3.0
scp>=0.14
pyyaml>=6.0
```

Install: `pip install paramiko scp pyyaml`

### Optional: Codex Agent (for autonomous optimization)
- Codex CLI (for agent-based candidate generation)
- OpenAI API access

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url> aka-local
cd aka-local

# 2. Create virtual environment
python -m venv .venv
.venv\\Scripts\\activate   # Windows
# source .venv/bin/activate  # Linux

# 3. Install dependencies
pip install paramiko scp pyyaml

# 4. Configure V100 access
# Edit config/environments/v100.yaml:
#   remote:
#     host: "YOUR_V100_IP"
#     user: "YOUR_SSH_USER"
#
# Set SSH password:
echo "YOUR_PASSWORD" > config/environments/.v100_secret
# OR set environment variable:
# set AKA_V100_PASSWORD=YOUR_PASSWORD   # Windows
# export AKA_V100_PASSWORD="YOUR_PASSWORD"     # Linux
```

---

## Configuration

All configuration is in `config/environments/v100.yaml`:

```yaml
environment: v100
gpu: Tesla V100-PCIE-16GB
architecture: sm_70
cuda_version: "11.8"

remote:
  host: "YOUR_V100_IP"      # REQUIRED: change this
  user: "YOUR_SSH_USER"     # REQUIRED: change this

paths:
  evaluator_dir: "~/cuda_kernel_experiments/evaluator"
  eval_script: "~/cuda_kernel_experiments/evaluator/eval.sh"
  work_dir: "~/aka_remote_jobs"

build:
  compiler: "nvcc"
  arch_flag: "-gencode arch=compute_70,code=sm_70"
  optimization: "-O2"

contract:
  function: "launch_kernel"
  signature: "void launch_kernel(float* x, float* weight, float* y, int batch, int hidden, float eps)"
  source_file: "candidate.cu"

evaluation:
  timeout_compile_s: 30
  timeout_total_s: 300
```

Evaluation shapes are configured in `config/environments/v100_sm70/evaluation.json`:

```json
{
  "shapes": [[1, 4096], [4, 4096], [8, 4096], [32, 4096]],
  "score": "geometric_mean_speedup",
  "correctness_tolerance": 0.001,
  "warmup": 50,
  "iterations": 200
}
```

---

## First Successful Run

### Step 1: Verify connectivity

```bash
python -m lab.cli doctor --env v100
```

Expected output:
```
[doctor] Checking environment: v100
[1/5] SSH connectivity ... [PASS]
[2/5] GPU check ... [PASS]
[3/5] CUDA compiler ... [PASS]
[4/5] Evaluator availability ... [PASS]
[5/5] Local knowledge integrity ... [PASS]
[doctor] Environment v100 (v100_sm70): READY
```

### Step 2: List available operators

```bash
python -m lab.cli operators
```

### Step 3: Run one optimization episode

```bash
python -m lab.cli run --env v100 --op rms_norm_v100_cuda --episodes 1
```

Expected output:
```
Phase 9: Unattended Campaign
  Operator:  rms_norm_v100_cuda
  Environment: v100
  Episodes:  1
  --- Ep N/1 ---
  Phase 1: AGENT
  Agent complete
  Phase 2: EVALUATION
  Result: ACCEPT score=X.XXX
  === Done: 1 ok, 0 failed ===
```

### Step 4: View report

```bash
python -m lab.cli report --env v100 --op rms_norm_v100_cuda
```

### Step 5: Replay verification

```bash
python -m lab.cli replay --env v100 --op rms_norm_v100_cuda --ep N
```

---

## Directory Structure

```
aka-local/
  config/environments/     # Environment configs
  operators/               # Operator definitions and references
  campaigns/               # Experiment campaigns (episodes, lineage)
  knowledge/environments/  # Per-environment knowledge (V100/RTX5060 isolated)
  lab/                     # CLI, evaluators, runtime, tools
  docs/                    # Documentation
```

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| SSH connection fails | Verify V100 IP, user, password in config |
| nvcc not found | Ensure CUDA 11.8 installed at /usr/local/cuda-11.8 |
| eval.sh missing | Clone evaluator to ~/cuda_kernel_experiments/evaluator |
| Compile fails | Verify nvcc supports sm_70 |
| Correctness fails | Check candidate.cu matches contract signature |
