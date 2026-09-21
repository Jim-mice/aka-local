"""Run-scoped Windows process ownership, termination and durable registry."""
from __future__ import annotations
import json, os, signal, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from ..core.persistence import atomic_json, read_json

def now(): return datetime.now(timezone.utc).isoformat()
def _summary(args): return " ".join(str(x) for x in list(args)[:3])

class ProcessRegistry:
    def __init__(self, run_dir, run_id, event_sink=None):
        self.run_dir=Path(run_dir);self.run_id=run_id;self.path=self.run_dir/"processes.json";self.event_sink=event_sink;self.run_dir.mkdir(parents=True,exist_ok=True);self.data=read_json(self.path,{"run_id":run_id,"processes":[]}) or {"run_id":run_id,"processes":[]};self._save()
    def _emit(self,t,p):
        if self.event_sink:self.event_sink(t,p)
    def _save(self): atomic_json(self.path,self.data)
    def attach(self, proc, *, kind, metadata=None):
        record={"process_id":f"{self.run_id}:{getattr(proc,'pid',None)}:{len(self.data['processes'])}","run_id":self.run_id,"kind":kind,"pid":getattr(proc,"pid",None),"parent_pid":os.getpid(),"command_summary":_summary(getattr(proc,"args",[])),"started_at":now(),"state":"ACTIVE","platform":"windows","process_group":"NEW_PROCESS_GROUP","metadata":metadata or {}}
        self.data["processes"].append(record);self._save();self._emit("PROCESS_STARTED",record);return record
    def completed(self,pid,returncode):
        for r in reversed(self.data["processes"]):
            if r["pid"]==pid and r["state"]=="ACTIVE":r.update(state="EXITED",returncode=returncode,ended_at=now());self._save();self._emit("PROCESS_EXITED",r);return r
    def active(self): return [r for r in self.data["processes"] if r.get("state")=="ACTIVE"]
    def terminate_tree(self, grace_s=1.0):
        results=[]
        for r in list(self.active()):
            pid=r.get("pid")
            if not pid:continue
            # Do not infer process ownership after restart: only registered PID
            # records for this run are targeted, and taskkill gets an arg list.
            try:
                subprocess.run(["taskkill","/PID",str(pid),"/T"],capture_output=True,text=True,timeout=grace_s)
                time.sleep(0.05)
                force=subprocess.run(["taskkill","/PID",str(pid),"/T","/F"],capture_output=True,text=True,timeout=5)
                r.update(state="FORCE_KILLED",ended_at=now(),termination_returncode=force.returncode);self._emit("PROCESS_FORCE_KILLED",r);results.append({"pid":pid,"returncode":force.returncode})
            except Exception as exc:r.update(state="TERMINATION_ERROR",error=f"{type(exc).__name__}: {exc}",ended_at=now());results.append({"pid":pid,"error":str(exc)})
        self._save();return results
    def mark_stale_after_restart(self):
        # PID reuse cannot be proven safe with the persisted record alone.
        # Mark it stale; recovery never kills it.
        changed=[]
        for r in self.active():r["state"]="STALE_PROCESS_RECORD";r["checked_at"]=now();changed.append(r);self._emit("STALE_PROCESS_RECORD",r)
        if changed:self._save()
        return changed

class ManagedProcessLauncher:
    def __init__(self, registry): self.registry=registry
    def run(self,args,*,kind,cwd=None,env=None,timeout=None,on_tick=None):
        flags=getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0)|getattr(subprocess,"CREATE_NO_WINDOW",0)
        proc=subprocess.Popen(args,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8",errors="replace",creationflags=flags)
        self.registry.attach(proc,kind=kind)
        started=time.monotonic()
        while True:
            remaining=None if timeout is None else max(0.0,timeout-(time.monotonic()-started))
            if timeout is not None and remaining<=0:
                self.registry.terminate_tree();raise subprocess.TimeoutExpired(args,timeout)
            try:
                out,err=proc.communicate(timeout=min(.25,remaining) if remaining is not None else .25);break
            except subprocess.TimeoutExpired:
                if on_tick:on_tick()
        self.registry.completed(proc.pid,proc.returncode);return proc.returncode,out,err
