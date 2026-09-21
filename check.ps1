. "$PSScriptRoot\env.ps1"
Write-Host "--- GPU ---"
nvidia-smi
Write-Host "--- CUDA ---"
nvcc --version
ncu --version
Write-Host "--- PyTorch ---"
& "$PSScriptRoot\.venv\Scripts\python.exe" -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())"
Write-Host "--- Codex ---"
codex --version
