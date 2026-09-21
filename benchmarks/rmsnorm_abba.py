import importlib.util
import json
import math
import statistics
import time
import argparse
from pathlib import Path

import torch


ROOT = Path(r"<PROJECT_ROOT>")
ATREX = Path(r"<LOCAL_USER_HOME>\projects\atrex-bench")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def event_ms(fn, reps):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    ref = load_module(args.baseline, "rms_ref")
    inp = load_module(ATREX / "data" / "rms_norm" / "input.py", "rms_input")
    cand = load_module(args.candidate, "rms_candidate")
    shapes = json.loads((ATREX / "data" / "rms_norm" / "shapes.json").read_text())
    rows = []
    ref_model = None
    cand_model = None
    for key in sorted(shapes, key=lambda x: int(x)):
        spec = shapes[key]
        kw = spec["input_kwargs"]
        init = spec["init_kwargs"]
        inputs = inp._make_inputs(**kw)
        ref_model = ref.Model(**init).cuda().eval()
        cand_model = cand.Model(**init).cuda().eval()
        with torch.inference_mode():
            expected = ref_model(**inputs)
            actual = cand_model(**inputs)
            max_abs = (actual.float() - expected.float()).abs().max().item()
            max_rel = ((actual.float() - expected.float()).abs() / expected.float().abs().clamp_min(1e-12)).max().item()
            for _ in range(args.warmup):
                ref_model(**inputs); cand_model(**inputs)
            torch.cuda.synchronize()
            a1 = event_ms(lambda: ref_model(**inputs), args.repeats)
            b1 = event_ms(lambda: cand_model(**inputs), args.repeats)
            b2 = event_ms(lambda: cand_model(**inputs), args.repeats)
            a2 = event_ms(lambda: ref_model(**inputs), args.repeats)
        a = (a1 + a2) / 2.0
        b = (b1 + b2) / 2.0
        rows.append({"shape_id": int(key), "token_count": kw["token_count"], "hidden_size": kw["hidden_size"], "A1_ms": a1, "B1_ms": b1, "B2_ms": b2, "A2_ms": a2, "incumbent_mean_ms": a, "candidate_mean_ms": b, "speedup": a / b if b else None, "max_abs": max_abs, "max_rel": max_rel})
        del inputs, expected, actual
        torch.cuda.empty_cache()
    speeds = [r["speedup"] for r in rows]
    out = {"operator": "rms_norm", "baseline": "official_eager_reference", "candidate": "luna_episode_1", "warmup": 20, "repetitions": 100, "ordering": "A/B/B/A", "rows": rows, "arithmetic_mean_speedup": statistics.mean(speeds), "geometric_mean_speedup": math.prod(speeds) ** (1.0 / len(speeds)), "total_time_ratio": sum(r["incumbent_mean_ms"] for r in rows) / sum(r["candidate_mean_ms"] for r in rows), "gpu": torch.cuda.get_device_name(), "arch": torch.cuda.get_device_capability()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("arithmetic_mean_speedup", "geometric_mean_speedup", "total_time_ratio", "gpu", "arch")}, indent=2))


if __name__ == "__main__":
    main()
