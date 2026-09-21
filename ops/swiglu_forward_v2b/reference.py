import torch


class Model(torch.nn.Module):
    def __init__(self, rows: int, cols: int, dtype: str = "float16"):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.dtype = getattr(torch, dtype)

    def forward(self, gate, up):
        return torch.nn.functional.silu(gate) * up
