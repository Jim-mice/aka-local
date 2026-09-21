"""Create a separately labeled, approved-by-test LOCAL smoke candidate snapshot."""
from __future__ import annotations
import json, shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CAMPAIGN=ROOT/"lab"/"campaigns"/"rms_norm_train__rtx5060_sm120__cuda_cpp"
SOURCE=ROOT/"ops"/"rms_norm_v2"
TARGET=CAMPAIGN/"episodes"/"e0003_luna_smoke_full_evaluator"/"candidate"

def main():
    if TARGET.exists():
        print(f"SMOKE_WORKBENCH_EXISTS {TARGET}");return
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(SOURCE,TARGET,ignore=shutil.ignore_patterns(".git","__pycache__","build","dist"))
    (TARGET.parent/"episode.json").write_text(json.dumps({"episode_id":"e0002_luna_smoke","working_directory":str(TARGET),"source_directory":str(SOURCE),"purpose":"user-approved one-experiment Luna integration smoke","created_at":datetime.now(timezone.utc).isoformat()},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"SMOKE_WORKBENCH_CREATED {TARGET}")
if __name__=="__main__":main()
