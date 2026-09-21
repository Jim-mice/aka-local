# Setup and safe reproduction

## Local Windows environment

Use a Python virtual environment outside version control. The exact historical package lock was not captured, so this repository does not claim reproducible pinned versions. Inspect imports for the workflow you intend to use; common optional dependencies include PyTorch, PyYAML, and Paramiko. CUDA workflows additionally need an NVCC toolchain compatible with the target configuration.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install only the dependencies required by the selected command. Do not treat this setup document as authorization to run a GPU campaign.

## Non-destructive inspection

These commands inspect existing state and do not launch an Agent campaign or GPU benchmark:

```powershell
py -3 -m json.tool .\five_target_campaign_state.json
py -3 -m json.tool .\real_target_campaign_index.json
.\lab.ps1 status
.\lab.ps1 list-ops
```

Review `lab.ps1` and `lab/cli.py --help` before using `run`, `evaluate`, or replay commands. Some historical target scripts are target-specific; the closure report records this reproducibility limitation.

## Optional remote V100 configuration

Copy `config/environments/v100.example.yaml` to `config/environments/v100.yaml`, then set only your own host, user, and remote paths. `v100.yaml` is ignored by Git. Use SSH keys, an interactive prompt, or a local ignored secret mechanism; do not put passwords on command lines or in environment files that may be captured in logs.

The example retains public scientific constraints (Tesla V100, CUDA 11.8, and `sm_70`) but contains no usable endpoint. A local Megatron checkout is an external dependency and is never modified by this workbench.
