"""Machine-readable full-shape RMSNorm correctness worker.

It deliberately reuses the existing Atrex input/reference contract; it does
not implement new benchmark mathematics.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ATREX = ROOT.parent / "atrex-bench" / "data" / "rms_norm"

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--candidate",type=Path,required=True); parser.add_argument("--reference",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    out={"pass":False,"tests":0,"failures":[],"max_abs":None,"max_rel":None}
    try:
        import torch
        reference=load(args.reference,"lab_reference"); candidate=load(args.candidate,"lab_candidate"); inputs=load(ATREX/"input.py","lab_input"); shapes=json.loads((ATREX/"shapes.json").read_text(encoding="utf-8"))
        max_abs=max_rel=0.0
        for key in sorted(shapes,key=lambda x:int(x)):
            spec=shapes[key]; data=inputs._make_inputs(**spec["input_kwargs"]); ref=reference.Model(**spec["init_kwargs"]).cuda().eval(); cand=candidate.Model(**spec["init_kwargs"]).cuda().eval()
            with torch.inference_mode():
                expected=ref(**data); actual=cand(**data); diff=(actual.float()-expected.float()).abs(); absolute=diff.max().item(); relative=(diff/expected.float().abs().clamp_min(1e-12)).max().item()
            out["tests"]+=1; max_abs=max(max_abs,absolute); max_rel=max(max_rel,relative)
            # Existing V2 historical contract permits bfloat16-scale errors.
            if not torch.allclose(actual.float(),expected.float(),rtol=1e-2,atol=2e-2): out["failures"].append({"shape_id":int(key),"max_abs":absolute,"max_rel":relative})
            del data,ref,cand,expected,actual
            torch.cuda.empty_cache()
        out.update(pass_=not out["failures"],max_abs=max_abs,max_rel=max_rel); out["pass"]=out.pop("pass_")
    except Exception as exc:
        out["error"]=f"{type(exc).__name__}: {exc}"; out["traceback"]=traceback.format_exc()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps(out,ensure_ascii=False));return 0 if out["pass"] else 1
if __name__=="__main__":raise SystemExit(main())
