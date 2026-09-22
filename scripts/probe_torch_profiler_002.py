import traceback
import torch

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
try:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ):
        x = torch.randn((8, 8), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
    print("PROFILE_PASS")
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
    print("PROFILE_BLOCKED")
