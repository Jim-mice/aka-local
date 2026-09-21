import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OP = ROOT / "ops" / "swiglu_forward_v2b"
spec = importlib.util.spec_from_file_location("aka_swiglu_v2b", OP / "kernel.py")
candidate_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate_module
spec.loader.exec_module(candidate_module)
Model = candidate_module.Model


def ref(gate, up):
    return torch.nn.functional.silu(gate) * up


def time_one(fn, gate, up, warmup=20, repeats=100):
    for _ in range(warmup):
        fn(gate, up)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn(gate, up)
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000.0 / repeats


def abba(fn_a, fn_b, gate, up, repeats=100):
    order = [fn_a, fn_b, fn_b, fn_a]
    values = [time_one(fn, gate, up, warmup=20, repeats=repeats) for fn in order]
    return {"A1": values[0], "B1": values[1], "B2": values[2], "A2": values[3]}


candidate = Model(1, 4096).cuda()
results = {
    "gpu": torch.cuda.get_device_name(),
    "capability": list(torch.cuda.get_device_capability()),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "shapes": {},
}

for m in (256, 1024, 4096):
    gate = torch.randn((m, 4096), device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    expected = ref(gate, up)
    actual = candidate(gate, up)
    max_abs = (actual - expected).abs().max().item()
    max_rel = ((actual - expected).abs() / expected.abs().clamp_min(1e-6)).max().item()
    a = lambda x, y: ref(x, y)
    b = lambda x, y: candidate(x, y)
    abba_values = abba(a, b, gate, up)
    ref_us = (abba_values["A1"] + abba_values["A2"]) / 2.0
    cand_us = (abba_values["B1"] + abba_values["B2"]) / 2.0
    results["shapes"][f"m{m}_d4096"] = {
        "reference_us": ref_us,
        "candidate_us": cand_us,
        "speedup": ref_us / cand_us,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "abba_us": abba_values,
    }

print(json.dumps(results, indent=2))
