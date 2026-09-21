import torch


def _make_inputs(rows: int, cols: int, dtype: str = "float16"):
    dt = getattr(torch, dtype)
    return {
        "gate": torch.randn((rows, cols), device="cuda", dtype=dt),
        "up": torch.randn((rows, cols), device="cuda", dtype=dt),
    }
