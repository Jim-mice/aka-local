"""Durable per-run configuration and safe-point manifest."""
from __future__ import annotations
import os, uuid
from datetime import datetime, timezone
from pathlib import Path
from ..core.persistence import atomic_json, read_json
def now(): return datetime.now(timezone.utc).isoformat()
class RunStore:
    def __init__(self, root, run_id=None): self.root=Path(root);self.run_id=run_id or str(uuid.uuid4());self.dir=self.root/"runtime"/"runs"/self.run_id;self.dir.mkdir(parents=True,exist_ok=True)
    def write_run(self,data): atomic_json(self.dir/"run.json",{**data,"run_id":self.run_id,"owner_pid":os.getpid(),"created_at":data.get("created_at",now())})
    def save(self,**state):
        current=read_json(self.dir/"resume_manifest.json",{}) or {};current.update(state,run_id=self.run_id,owner_pid=os.getpid(),updated_at=now());atomic_json(self.dir/"resume_manifest.json",current);atomic_json(self.dir/"state.json",current);atomic_json(self.dir/"checkpoint.json",current);return current
    @staticmethod
    def recoverable(root):
        base=Path(root)/"runtime"/"runs";out=[]
        for p in base.glob("*/resume_manifest.json"):
            x=read_json(p,{}) or {}
            if x.get("state") in {"PAUSED","INTERRUPTED","RUNNING"}:out.append(x)
        return out
