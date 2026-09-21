"""List all available operators including V100 CUDA operators."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent

# Registry operators
reg = ROOT / "lab" / "registry" / "operators"
reg_ops = [p.stem for p in reg.glob("*.yaml")] if reg.is_dir() else []

# V100 CUDA operators
ops_dir = ROOT / "operators"
v100_ops = []
if ops_dir.is_dir():
    for p in ops_dir.iterdir():
        if p.is_dir() and (p / "metadata.json").is_file():
            v100_ops.append(p.name)

for op in sorted(set(reg_ops + v100_ops)):
    print(op)
