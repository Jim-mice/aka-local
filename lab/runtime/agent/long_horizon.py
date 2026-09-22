"""Controller-owned bounded LOCAL RTX5060 RMSNorm episode runner."""
from __future__ import annotations
import hashlib, json, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

from ...core.journal import append_experiment, append_experiment_with_hypothesis, read_experiments, read_hypotheses, sync_live
from ...core.memory import write_canonical
from ...core.persistence import atomic_json, read_json
from ...core.probe import probe_local
from ...core.normalization import get_record_backend_ids, get_record_operator_ids, get_record_platform_ids
from ...core.context_builder import build_authoritative_context
from ...core.experiment_accounting import account_experiments
from ..path_policy import CandidatePathPolicy, PathPolicyViolation
from ..workspace import CandidateWorkspace
from ..recovery import RecoveryManager
from ..supervisor.controller_policy import decide, finalize_after_robustness
from ...core.hypothesis import Hypothesis
from ...core.evidence import Evidence, EvidenceType, Verdict
from ...core.diagnostic import Diagnostic, DiagnosticCategory

from .episode_artifacts import archive_episode, write_knowledge_candidates
from .attempt_loop import HypothesisAttemptController, LineageStore, StructuredKnowledgeSink, load_attempt_budget

def utc(): return datetime.now(timezone.utc).isoformat()
def digest(root):
    h=hashlib.sha256(); root=Path(root)
    for p in sorted(root.rglob("*")):
        # Python imports create __pycache__ during read-only diagnostics.  It is
        # a derived artifact, not candidate source identity (same rule as the
        # Context Builder), and must not turn a completed measurement into
        # DIAGNOSTIC_CANDIDATE_CHANGED before the journal record is appended.
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts: h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()
class InvalidAgentOutput(ValueError): pass

class LongHorizonRunner:
    def __init__(self,*,campaign_dir,candidate_root,incumbent,session,evaluator,events,budget=None,workbench=None,lab_root=None,run_store=None):
        self.campaign_dir=Path(campaign_dir).resolve();self.candidate_root=Path(candidate_root).resolve();self.incumbent=Path(incumbent).resolve();self.session,self.evaluator,self.events=session,evaluator,events
        self.lab_root=Path(lab_root or self.campaign_dir.parents[1]).resolve();self.workbench=workbench or {};self.budget={"max_experiments":5,"max_consecutive_failures":3,**(budget or {})}
        self.episode_dir=self.candidate_root.parent;self.journal=self.episode_dir/"journal.jsonl";self.live=self.episode_dir/"live.json";self.pause_requested=False;self.stop_requested=False;self.run_store=run_store;self.recovery=RecoveryManager(self.run_store.dir,self.run_store.run_id) if run_store else None;self.workspace=CandidateWorkspace(self.episode_dir,self.incumbent,self.campaign_dir,(self.incumbent,self.campaign_dir/"memory",self.campaign_dir/"incumbent",self.lab_root/"knowledge",self.lab_root/"registry"))
        self.policy=CandidatePathPolicy(self.candidate_root,[self.incumbent,self.campaign_dir/"memory",self.campaign_dir/"incumbent",self.lab_root/"knowledge",self.lab_root/"registry"])
    def _emit(self,t,p=None): return self.events.append(t,p or {},workbench_id=self.workbench.get("workbench_id"))
    def build_hypothesis_attempt_controller(self, evaluator, *, lineage_path=None, knowledge_path=None, budget_path=None):
        """Create the P1 controller using this runner's persistent session.

        Callers use this for a single hypothesis when compile/correctness
        repairs must stay on the same Agent thread.  It intentionally does
        not grant the Agent evaluator, qualification, or promotion control.
        """
        policy_path = Path(budget_path or self.lab_root / "config" / "policies" / "agent_attempts.json")
        return HypothesisAttemptController(
            session=self.session, evaluator=evaluator,
            lineage=LineageStore(lineage_path or self.campaign_dir / "lineage.jsonl"),
            knowledge=StructuredKnowledgeSink(knowledge_path or self.campaign_dir / "knowledge_attempts.jsonl"),
            budget=load_attempt_budget(policy_path),
        )
    @staticmethod
    def _logical_number(identifier):
        match=re.search(r"-x(\d+)",str(identifier))
        return int(match.group(1)) if match else 1
    @staticmethod
    def _execution_action(action):
        """Drop Agent plan narrative metadata before invoking an adapter.

        ``scope`` and ``requirements`` explain intent to a human/Agent; they
        are not evaluator keyword arguments.  Profiler shape selections are the
        one structured planning field that maps to an executable argument:
        normalize ``requirements.shape_selection`` to ``shape_ids`` without
        changing measurement semantics.  A capability-discovery-only profiler
        action maps to a read-only capability probe rather than a zero-shape
        profile request.
        """
        action=dict(action or {});kind=action.get("type")
        base={"repeated_per_shape_benchmark":{"shapes","batches","warmup","repeats"},"profiler":{"shape_ids","questions","test_only"},"compare_regimes":{"repeated_summary"},"static_evidence":set(),"inspect_shape_metadata":set()}.get(kind,set())
        filtered={"type":kind,**{key:value for key,value in action.items() if key in base}}
        if kind=="profiler" and "shape_ids" not in filtered:
            requirements=action.get("requirements") or {}
            selection=requirements.get("shape_selection")
            if selection is not None:
                filtered["shape_ids"]=[str(item.get("shape_id") if isinstance(item,dict) else item) for item in selection]
            elif requirements.get("discover"):
                filtered["capability_only"]=True
        return filtered
    def _status(self,s,**p):
        self._emit("AGENT_STATUS",{"status":s,**p});sync_live(self.live,state=s,workbench=self.workbench,updated_at=utc())
        if self.run_store:self.run_store.save(state="RUNNING",last_safe_point=s,current_experiment_index=p.get("experiment"),journal_path=str(self.journal),live_path=str(self.live),candidate_root=str(self.candidate_root),candidate_hash=digest(self.candidate_root),environment_fingerprint=self.workbench.get("environment_fingerprint"),resume_safe=True)
    def pause(self): self.pause_requested=True;self._emit("PAUSE_REQUESTED")
    def stop(self): self.stop_requested=True;self._emit("STOP_REQUESTED")
    def _knowledge(self,n):
        self._status("INSPECTING_KNOWLEDGE",experiment=n);items=[]
        for p in sorted((self.lab_root/"knowledge").glob("**/*.json")):
            try:r=json.loads(p.read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError):continue
            if "rms_norm_train" in get_record_operator_ids(r) and "rtx5060_laptop_sm120" in get_record_platform_ids(r) and "cuda_cpp" in get_record_backend_ids(r):items.append({"id":r.get("id"),"title":r.get("title") or r.get("statement"),"type":r.get("type")})
        frontier=self.campaign_dir/"frontier.json";snap={"query":{"operator":"rms_norm_train","platform":"rtx5060_laptop_sm120","backend":"cuda_cpp","experiment":n},"timestamp":utc(),"knowledge_ids":[x["id"] for x in items],"knowledge":items,"canonical_memory_ids":[p.stem for p in sorted((self.campaign_dir/"memory").glob("v*.json"))[-5:]],"frontier_hash":hashlib.sha256(frontier.read_bytes()).hexdigest() if frontier.exists() else None}
        snap["snapshot_hash"]=hashlib.sha256(json.dumps(snap,sort_keys=True,ensure_ascii=False).encode()).hexdigest();atomic_json(self.episode_dir/"knowledge_snapshot.json",snap);atomic_json(self.episode_dir/f"knowledge_snapshot_e{n:03d}.json",snap);self._emit("KNOWLEDGE_REFRESHED",snap);return snap
    def _context(self,n,knowledge):
        """Use the same filesystem-derived context as Agent Console ASK."""
        c=build_authoritative_context(
            self.lab_root,self.campaign_dir,self.candidate_root,
            workbench=self.workbench,events=self.events,
            operator_id="rms_norm_train",platform_id="rtx5060_laptop_sm120",backend_id="cuda_cpp",
        )
        c["experiment"]=n;c["knowledge_snapshot"]=knowledge
        c["context_hash"]=hashlib.sha256(json.dumps(c,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
        atomic_json(self.episode_dir/"context_snapshot.json",c);atomic_json(self.episode_dir/f"context_snapshot_e{n:03d}.json",c)
        # A directive remains PENDING after its context is written.  It becomes
        # consumed only after the model has returned a valid, persisted plan.
        # That prevents a failed/context-only resume from silently consuming it.
        c["consumed_directive_ids"]=[]
        return c

        # Superseded by the shared builder above; retained only to avoid a
        # broader refactor in this focused context-wiring repair.
        directives=[e.get("payload",{}) for e in self.events.replay() if e.get("type")=="HUMAN_DIRECTIVE"][-10:]
        c={"campaign":read_json(self.campaign_dir/"campaign.json",{}) or {},"workbench":self.workbench,"candidate_root":str(self.candidate_root),"incumbent":str(self.incumbent),"knowledge":knowledge,"journal":str(self.journal),"frontier":read_json(self.campaign_dir/"frontier.json",{}),"human_directives":directives,"protected_paths":[str(x) for x in (self.incumbent,self.campaign_dir/"memory",self.campaign_dir/"incumbent",self.lab_root/"knowledge",self.lab_root/"registry")],"evaluator_capabilities":["compile","56-shape correctness","development same-process ABBA","authoritative same-process ABBA","repeated robustness"],"experiment":n}
        c["context_hash"]=hashlib.sha256(json.dumps(c,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest();atomic_json(self.episode_dir/"context_snapshot.json",c);atomic_json(self.episode_dir/f"context_snapshot_e{n:03d}.json",c);return c
    @staticmethod
    def _reply(result):
        text=str(getattr(result,"final_response",result));m=re.search(r"```json\s*(\{.*?\})\s*```",text,re.S)
        try:return json.loads(m.group(1) if m else text.strip())
        except json.JSONDecodeError as e:raise InvalidAgentOutput(f"INVALID_AGENT_OUTPUT: {e}") from e
    def _hypothesis_history(self, records):
        """Build a summary of previous hypotheses and their outcomes."""
        items = []
        for rec in records or []:
            plan = rec.get('plan') or {}
            hyp = plan.get('hypothesis') or {}
            claim = hyp.get('claim') or hyp.get('question')
            if not claim:
                continue
            decision = rec.get('decision', 'UNKNOWN')
            benchmark = rec.get('development_benchmark') or {}
            metrics = benchmark.get('metrics') or {}
            speedup = metrics.get('arithmetic_mean_speedup')
            items.append({
                'experiment_id': rec.get('experiment_id'),
                'claim': claim,
                'mechanism': hyp.get('mechanism', ''),
                'decision': decision,
                'speedup': speedup,
            })
        return items

    def _plan(self,c):
        hyp_history = self._hypothesis_history(read_experiments(self.journal))
        hyp_context = "HYPOTHESIS HISTORY:\n" + json.dumps(hyp_history, ensure_ascii=False, default=str) + "\n\n" if hyp_history else ""
        prompt="""Return ONLY a JSON object inside a ```json block. Do NOT edit in this turn. Required keys: experiment_kind (DIAGNOSTIC|OPTIMIZATION), diagnostic_actions (array; required for DIAGNOSTIC), supporting_evidence_ids (array; required for evidence-backed OPTIMIZATION), analysis_summary, bottleneck, evidence_summary, hypothesis {question,claim,rationale,expected_effect,support_condition,refute_condition}, planned_change, expected_risk, decision_request, next_direction. A DIAGNOSTIC action may be repeated_per_shape_benchmark, static_evidence, profiler, inspect_shape_metadata, or compare_regimes. DIAGNOSTIC never edits candidate or runs the standard compile/correctness/dev-ABBA pipeline. decision_request is keep_as_best, reject_and_continue, ready_for_gate, or blocked; never promotion/accept. Filesystem/context snapshot is authoritative. CONTEXT:\n"""+hyp_context+json.dumps(c,ensure_ascii=False,default=str)
        last_error=""
        for turn in range(2):
            instruction=prompt if not turn else "You selected a diagnostic experiment but did not provide executable diagnostic_actions. Return a valid structured diagnostic plan. Each diagnostic_actions item must be an object with a type field, not a string. Return ONLY the required JSON plan."
            r=self._reply(self.session.run_experiment_turn(instruction,c));need={"analysis_summary","bottleneck","evidence_summary","hypothesis","planned_change","expected_risk","decision_request","next_direction"}
            # Existing persisted/test scripted plans remain readable.  New
            # agents are asked for the explicit field above.
            legacy_kind="experiment_kind" not in r
            r.setdefault("experiment_kind","OPTIMIZATION");r.setdefault("diagnostic_actions",[]);r.setdefault("supporting_evidence_ids",[])
            base_valid=not (need-set(r)) and isinstance(r["hypothesis"],dict) and isinstance(r["hypothesis"].get("claim"),str) and len(r["hypothesis"].get("claim","").strip())>0 and r["decision_request"] in {"keep_as_best","reject_and_continue","ready_for_gate","blocked"} and r["experiment_kind"] in {"DIAGNOSTIC","OPTIMIZATION"}
            diagnostic_valid=r["experiment_kind"]!="DIAGNOSTIC" or (isinstance(r["diagnostic_actions"],list) and len(r["diagnostic_actions"])>=1 and all(isinstance(action,dict) and isinstance(action.get("type"),str) and action["type"] in {"repeated_per_shape_benchmark","static_evidence","profiler","inspect_shape_metadata","compare_regimes"} for action in r["diagnostic_actions"]))
            if base_valid and diagnostic_valid:
                r["_legacy_plan"]=legacy_kind;return r
            if isinstance(r.get("hypothesis"),dict) and not r["hypothesis"].get("claim","").strip():
                last_error="MISSING_HYPOTHESIS"
            else:
                last_error="INVALID_DIAGNOSTIC_PLAN" if r.get("experiment_kind")=="DIAGNOSTIC" else "INVALID_AGENT_OUTPUT"
        raise InvalidAgentOutput(last_error or "INVALID_AGENT_OUTPUT: structured plan missing")

    def _plan_path(self,n,attempt=1):
        suffix="" if attempt==1 else f"_r{attempt-1}"
        return self.episode_dir/f"validated_plan_e{n:03d}{suffix}.json"

    def _persist_validated_plan(self,n,plan,context,attempt=1):
        """Persist only a fully validated executable plan at a safe boundary."""
        payload={"episode_id":self.episode_dir.name,"experiment_id":f"{self.episode_dir.name}-x{n:03d}",
                 "attempt":attempt,"created_at":utc(),"context_hash":context.get("context_hash"),"plan":plan}
        atomic_json(self._plan_path(n,attempt),payload)
        return str(self._plan_path(n,attempt))

    def _load_validated_plan(self,n,context,attempt=1):
        """Use a prior safe plan on recovery; never treat a raw event as a plan."""
        payload=read_json(self._plan_path(n,attempt),{}) or {}
        if payload.get("plan") is None and attempt>1:
            # A framework retry executes the same already-validated logical
            # plan.  Prefer the newest prior attempt instead of asking the
            # model to recreate an evidence-backed plan from memory.
            for prior in range(attempt-1,0,-1):
                payload=read_json(self._plan_path(n,prior),{}) or {}
                if payload.get("plan") is not None:break
        plan=payload.get("plan")
        if not isinstance(plan,dict):
            return None
        try:
            # Reuse the same strict contract without asking a model.
            kind=plan.get("experiment_kind")
            actions=plan.get("diagnostic_actions",[])
            valid=kind in {"DIAGNOSTIC","OPTIMIZATION"}
            if kind=="DIAGNOSTIC":
                valid=valid and isinstance(actions,list) and bool(actions) and all(
                    isinstance(action,dict) and action.get("type") in {
                        "repeated_per_shape_benchmark","static_evidence","profiler",
                        "inspect_shape_metadata","compare_regimes"
                    } for action in actions)
            if not valid:
                return None
            # A context hash mismatch is expected after recovery/knowledge refresh;
            # the persisted plan is still valid only as an execution plan, while the
            # new context remains authoritative for its evidence and interpretation.
            return plan
        except Exception:
            return None

    def _consume_directives_at_plan(self,context,n):
        consumed=[]
        for directive in context.get("active_human_directives",[]):
            directive_id=directive.get("directive_id")
            if directive_id:
                self._emit("HUMAN_DIRECTIVE_STATUS",{
                    "target_event_id":directive_id,"status":"CONSUMED",
                    "phase":"CONSUMED_AT_PLAN","episode_id":self.episode_dir.name,
                    "experiment_id":f"{self.episode_dir.name}-x{n:03d}",
                })
                consumed.append(directive_id)
        return consumed

    def _run_diagnostics(self, plan, context, n):
        """Execute only controller-owned diagnostic actions; never edit code."""
        results=[];last_repeated=None
        self._heartbeat("PROFILE", n)
        self._emit("DIAGNOSTIC_STARTED",{"experiment":n,"actions":plan.get("diagnostic_actions",[])})
        for action in plan.get("diagnostic_actions",[]):
            action=dict(action or {});kind=action.get("type")
            if kind not in {"repeated_per_shape_benchmark","static_evidence","profiler","inspect_shape_metadata","compare_regimes"}:
                results.append({"action":action,"pass":False,"reason":"UNKNOWN_DIAGNOSTIC_ACTION"});continue
            if kind=="compare_regimes" and last_repeated:
                action["repeated_summary"]=(last_repeated.get("metrics") or last_repeated)
            self._status("RUNNING_DIAGNOSTIC",experiment=n,action=kind);self._emit("DIAGNOSTIC_ACTION_STARTED",{"experiment":n,"action":action})
            def progress(payload): self._emit("DIAGNOSTIC_PROGRESS",{"experiment":n,"action":kind,**(payload or {})})
            execution_action=self._execution_action(action)
            try:
                result=self.evaluator.run_diagnostic_action(execution_action,progress_callback=progress)
            except Exception as exc:
                result={"phase":kind,"pass":False,"status":"DIAGNOSTIC_ACTION_FAILED",
                        "reason":f"{type(exc).__name__}: {exc}","framework_execution_failure":True}
                self._emit("DIAGNOSTIC_ACTION_FAILED",{"experiment":n,"action":kind,"result":result})
            if kind=="repeated_per_shape_benchmark":last_repeated=result
            entry={"action":action,"result":result}
            # Phase 3: convert profiler results to structured Diagnostics
            try:
                if kind == "profiler" and isinstance(result, dict):
                    parsed = result.get("parsed") or result
                    diags = Diagnostic.from_profile(parsed, f"{self.episode_dir.name}-x{n:03d}")
                    entry["diagnostics"] = [d.to_dict() for d in diags]
                elif kind == "repeated_per_shape_benchmark" and isinstance(result, dict):
                    diags = Diagnostic.from_benchmark_summary(result, f"{self.episode_dir.name}-x{n:03d}")
                    entry["diagnostics"] = [d.to_dict() for d in diags]
            except Exception:
                pass
            results.append(entry)
            event="PROFILE_CAPABILITY" if kind=="profiler" else "STATIC_EVIDENCE_COLLECTED" if kind=="static_evidence" else "DIAGNOSTIC_ACTION_RESULT"
            self._emit(event,{"experiment":n,"action":kind,"result":result})
            artifacts=(result or {}).get("artifacts") or {}
            if artifacts:self._emit("DIAGNOSTIC_EVIDENCE_WRITTEN",{"experiment":n,"action":kind,"artifacts":artifacts})
        return results

    @staticmethod
    def _diagnostic_framework_failure(results):
        for entry in results or []:
            result=(entry or {}).get("result") or {}
            if result.get("framework_execution_failure"):
                return result
        return None

    def _interpret_diagnostic(self, record, context):
        self._status("INTERPRETING_RESULT",experiment=record["experiment_id"])
        evidence={"diagnostic_results":record.get("diagnostic_results",[]),"experiment_id":record["experiment_id"]}
        prompt="Return ONLY JSON in a ```json block: decision_request (keep_as_best|reject_and_continue|ready_for_gate|blocked), result_summary, next_direction. This was a DIAGNOSTIC experiment: do not claim promotion and do not claim code performance gain without changed candidate evidence. Evidence: "+json.dumps(evidence,ensure_ascii=False,default=str)
        advisory=self._reply(self.session.run_experiment_turn(prompt,context))
        if advisory.get("decision_request") not in {"keep_as_best","reject_and_continue","ready_for_gate","blocked"}:raise InvalidAgentOutput("INVALID_AGENT_DECISION")
        if advisory.get("decision_request")=="ready_for_gate":
            raise InvalidAgentOutput("INVALID_AGENT_DECISION: DIAGNOSTIC cannot request supervisor gate")
        return advisory

    def _optimization_evidence_ready(self, plan, context):
        if plan.get("_legacy_plan") or plan.get("supporting_evidence_ids"):
            return True
        directives=" ".join(str(item.get("text", "")) for item in context.get("active_human_directives",[])).lower()
        return "explor" in directives or "探索" in directives
    def _edit(self,plan,c):
        self._status("EDITING_CANDIDATE");prompt=f"""Implement this plan. You may modify ONLY ACTIVE WORKBENCH ROOT. Do not run compile, correctness, benchmark, profile, git, network commands.

MANDATORY WRITE SAFETY: preserve the complete existing content of every file. Do NOT replace, truncate, delete, or recreate a source file. Apply the smallest possible in-place diff. If you cannot make a safe minimal edit, return changed_files=[] and explain why. The controller will reject any source file that loses its existing non-comment lines.

Return ONLY JSON in a ```json block: changed_files (relative paths array), diff_summary, build_expectation. ACTIVE WORKBENCH ROOT: {self.candidate_root}\nPROTECTED: {json.dumps(c['protected_paths'])}\nPLAN: {json.dumps(plan,ensure_ascii=False)}"""
        r=self._reply(self.session.run_experiment_turn(prompt,c))
        if not isinstance(r.get("changed_files"),list):raise InvalidAgentOutput("INVALID_AGENT_OUTPUT: changed_files absent")
        for rel in r["changed_files"]:self.policy.require_allowed(self.candidate_root/rel)
        return r
    def _candidate_lineage(self,record):
        """Persist deterministic source identity for each edited candidate."""
        plan=record.get("plan") or {};hypothesis=plan.get("hypothesis") or {}
        hypothesis_id=str(plan.get("hypothesis_id") or hypothesis.get("id") or ("hyp-"+hashlib.sha256(json.dumps(hypothesis,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()[:16]))
        source_hash=digest(self.candidate_root)
        prior=[]
        path=self.episode_dir/"candidate_lineage.jsonl"
        if path.exists():
            prior=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        incumbent_id="incumbent:"+hashlib.sha256(json.dumps({"name":self.campaign_dir.name,"source":str(self.incumbent),"hash":digest(self.incumbent)},sort_keys=True).encode()).hexdigest()[:16]
        parent=(prior[-1] if prior else {}).get("candidate_id") or incumbent_id
        candidate_id="candidate:"+hashlib.sha256(json.dumps({"campaign":self.campaign_dir.name,"episode":self.episode_dir.name,"experiment":record["experiment_id"],"source_hash":source_hash},sort_keys=True).encode()).hexdigest()[:16]
        item={"candidate_id":candidate_id,"parent_candidate_id":parent,"incumbent_id":incumbent_id,"episode_id":self.episode_dir.name,"experiment_id":record["experiment_id"],"source_hash":source_hash,"changed_files":record.get("changed_files") or [],"hypothesis_id":hypothesis_id,"created_at":utc()}
        with path.open("a",encoding="utf-8") as f:f.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+"\n")
        record["candidate_lineage"]=item;return item
    def _commit(self,r):
        if not (self.campaign_dir/".git").exists():return None
        subprocess.run(["git","-C",str(self.campaign_dir),"add","--",str(self.candidate_root.relative_to(self.campaign_dir))],capture_output=True,text=True)
        subprocess.run(["git","-C",str(self.campaign_dir),"commit","-m",f"episode {r['episode_id']} selected {r['experiment_id']}"],capture_output=True,text=True)
        h=subprocess.run(["git","-C",str(self.campaign_dir),"rev-parse","HEAD"],capture_output=True,text=True);return h.stdout.strip() if not h.returncode else None
    def _heartbeat(self, phase, experiment=0, *, experiment_id="", candidate_id=""):
        """Update recovery heartbeat and emit HEARTBEAT_UPDATED event."""
        if self.recovery:
            self.recovery.update_heartbeat(phase, experiment, experiment_id=experiment_id, candidate_id=candidate_id)
            self._emit("HEARTBEAT_UPDATED", {
                "run_id": self.recovery.run_id,
                "phase": phase,
                "timestamp": utc(),
                "experiment_id": experiment_id,
            })

    def _finish(self,state,records,**extra):
        accounting=account_experiments(records,self.events.replay())
        self._heartbeat("DONE", accounting.get("valid_experiment_count", 0))
        sync_live(self.live,state=state,episode=str(self.episode_dir),experiments=accounting["valid_experiment_count"],framework_attempts=accounting["framework_attempt_count"],environment_fingerprint=self.workbench.get("environment_fingerprint"),**extra)
        if self.run_store:self.run_store.save(state=state,last_safe_point="AFTER_SUPERVISOR" if state in {"PROMOTE","REJECT_PERFORMANCE","REJECT_CORRECTNESS","REJECT_COMPILE"} else "AFTER_EXPERIMENT",current_experiment_index=accounting["valid_experiment_count"],journal_path=str(self.journal),live_path=str(self.live),candidate_root=str(self.candidate_root),candidate_hash=digest(self.candidate_root),resume_safe=True,**extra)
        return {"state":state,"experiments":records,"valid_experiment_count":accounting["valid_experiment_count"],"framework_attempt_count":accounting["framework_attempt_count"],**extra}
    def run(self):
        self.policy.require_allowed(self.candidate_root);self.workspace.create();self._emit("CANDIDATE_WORKSPACE_CREATED",{"candidate_root":str(self.workspace.candidate_root),"mode":self.workspace.mode})
        if self.recovery:
            self._heartbeat("START", 0)
        prep = self.evaluator.prepare()
        if not prep.get("pass"):return {"state":"BLOCKED","reason":"ENVIRONMENT_NOT_READY","prepare":prep}
        probe=probe_local()
        if probe.get("status")!="READY":return {"state":"BLOCKED","reason":"ENVIRONMENT_CHANGED","probe":probe}
        self.workbench["environment_fingerprint"]=probe.get("environment_fingerprint");self.session.start()
        if self.recovery:
            self._heartbeat("RUN_START", 0)
        self._emit("AGENT_SESSION_STARTED",{"model":getattr(self.session,"model","unknown"),"reasoning_effort":getattr(self.session,"effort","unknown")});records=read_experiments(self.journal);fails=0
        try:
          while True:
            accounting=account_experiments(records,self.events.replay())
            if accounting["valid_experiment_count"]>=int(self.budget["max_experiments"]):return self._finish("BUDGET_EXHAUSTED",records)
            slot=accounting["next_slot"]
            if slot.get("blocked"):return self._finish("WAITING_FOR_HUMAN",records,reason=slot.get("reason","FRAMEWORK_RETRY_LIMIT_REACHED"))
            n=self._logical_number(slot["logical_experiment_id"]);attempt=slot["attempt"];logical_id=slot["logical_experiment_id"]
            if "-x" not in logical_id:logical_id=f"{self.episode_dir.name}-{logical_id}"
            experiment_id=logical_id if attempt==1 else f"{logical_id}-r{attempt-1}"
            if self.stop_requested:return self._finish("STOPPED",records)
            if self.pause_requested:return self._finish("PAUSED",records)
            self._emit("EXPERIMENT_STARTED",{"number":n,"experiment_id":experiment_id,"logical_experiment_id":logical_id,"attempt":attempt})
            self._heartbeat("PLAN", n)
            knowledge=self._knowledge(n);context=self._context(n,knowledge);self.session.send_context(context);self._status("ANALYZING_INCUMBENT",experiment=n)
            try:
                plan=self._load_validated_plan(n,context,attempt)
                plan_source="RECOVERED_VALIDATED_PLAN" if plan else "AGENT_PLAN"
                if not plan:
                    plan=self._plan(context)
                    self._persist_validated_plan(n,plan,context,attempt)
                    self._consume_directives_at_plan(context,n)
                plan_payload={"experiment":n,"experiment_id":experiment_id,"logical_experiment_id":logical_id,"attempt":attempt,"plan":plan,"plan_source":plan_source,"plan_path":str(self._plan_path(n,attempt))}
                self._emit("DIAGNOSTIC_PLAN_READY" if plan["experiment_kind"]=="DIAGNOSTIC" else "OPTIMIZATION_PLAN_READY",plan_payload);self._emit("HYPOTHESIS_RECORDED",plan);self._heartbeat("PLAN", n, experiment_id=experiment_id, candidate_id="")
                ws_snap=self.workspace.snapshot();self._emit("CANDIDATE_SNAPSHOT_CREATED",ws_snap);before=self.policy.snapshot(self.workspace.candidate_root);candidate_digest_before=digest(self.workspace.candidate_root)
                rec={"experiment_id":experiment_id,"logical_experiment_id":logical_id,"attempt":attempt,"episode_id":self.episode_dir.name,"campaign_id":self.campaign_dir.name,"timestamp":utc(),"workbench_id":self.workbench.get("workbench_id"),"working_directory":str(self.candidate_root),"source_directory":str(self.candidate_root),"incumbent_directory":str(self.incumbent),"environment_fingerprint":self.workbench.get("environment_fingerprint"),"candidate_hash_before":hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),"candidate_hash":digest(self.candidate_root),"plan":plan,"experiment_kind":plan["experiment_kind"],"knowledge":knowledge,"context_hash":context["context_hash"]}
                if plan["experiment_kind"]=="DIAGNOSTIC":
                    rec["diagnostic_actions"]=plan.get("diagnostic_actions",[])
                    rec["diagnostic_results"]=self._run_diagnostics(plan,context,n)
                    rec["candidate_hash"]=digest(self.candidate_root);rec["changed_files"]=[]
                    if rec["candidate_hash"]!=candidate_digest_before: raise PathPolicyViolation("DIAGNOSTIC_CANDIDATE_CHANGED")
                    framework_failure=self._diagnostic_framework_failure(rec["diagnostic_results"])
                    if framework_failure:
                        rec["decision"]="INVALID_FRAMEWORK_FAILURE";rec["framework_failure"]=framework_failure
                    else:
                        advisory=self._interpret_diagnostic(rec,context);rec["agent_interpretation"]=advisory;rec["decision"]=advisory["decision_request"].upper();fails=0
                else:
                    if not self._optimization_evidence_ready(plan,context):
                        return self._finish("WAITING_FOR_HUMAN",records,reason="MISSING_SUPPORTING_EVIDENCE")
                    source_guard=self.policy.source_line_guard();edit=self._edit(plan,context);after=self.policy.snapshot(self.candidate_root);changed=self.policy.validate_snapshot_delta(before,after);self.policy.validate_source_line_guard(source_guard)
                    rec.update({"candidate_hash":digest(self.candidate_root),"agent_edit":edit,"changed_files":changed});self._candidate_lineage(rec);self._heartbeat("EDIT", n,experiment_id=experiment_id)
                    self._status("COMPILING",experiment=n);self._heartbeat("BUILD", n,experiment_id=experiment_id);self._emit("COMPILE_STARTED",{"experiment":n});rec["compile"]=self.evaluator.compile();self._emit("COMPILE_RESULT",rec["compile"]);self._heartbeat("BUILD", n, experiment_id=experiment_id)
                    if not rec["compile"].get("pass"):rec["decision"]="REJECT_COMPILE";fails+=1;self.workspace.reset_to_baseline();self._emit("CANDIDATE_RESET",{"reason":"REJECT_COMPILE","experiment":n})
                    else:
                        self._status("CHECKING_CORRECTNESS",experiment=n);self._heartbeat("CORRECTNESS", n,experiment_id=experiment_id);self._emit("CORRECTNESS_STARTED",{"experiment":n});rec["correctness"]=self.evaluator.check_correctness();self._emit("CORRECTNESS_RESULT",rec["correctness"]);self._heartbeat("CORRECTNESS", n, experiment_id=experiment_id)
                        if not rec["correctness"].get("pass"):rec["decision"]="REJECT_CORRECTNESS";fails+=1;self.workspace.reset_to_baseline();self._emit("CANDIDATE_RESET",{"reason":"REJECT_CORRECTNESS","experiment":n})
                        else:
                            self._status("RUNNING_DEV_BENCHMARK",experiment=n);self._heartbeat("BENCHMARK", n,experiment_id=experiment_id);self._emit("BENCHMARK_STARTED",{"experiment":n,"method":"DEVELOPMENT_ABBA"});rec["development_benchmark"]=self.evaluator.development_benchmark();self._emit("BENCHMARK_RESULT",rec["development_benchmark"]);self._heartbeat("BENCHMARK", n, experiment_id=experiment_id);self._status("INTERPRETING_RESULT",experiment=n)
                        # Phase 2: evidence verification
                        try:
                            evidence = Evidence.from_evaluation(
                                {"compile":rec["compile"],"correctness":rec["correctness"],
                                 "authoritative_abba":rec["development_benchmark"]},
                                rec["experiment_id"])
                            rec["evidence"] = evidence.to_dict()
                            hyp = Hypothesis.from_dict(plan.get("hypothesis") or {})
                            rec["hypothesis_verdict"] = hyp.verify(rec.get("development_benchmark") or {})
                            rec["decision"] = "READY_FOR_GATE"
                            fails = 0
                            self._emit("HYPOTHESIS_VERIFIED",{"experiment_id":rec["experiment_id"],"verdict":rec["hypothesis_verdict"],"hypothesis":hyp.to_dict()})
                        except Exception:
                            rec["hypothesis_verdict"] = "INCONCLUSIVE"
                            instruction="Return ONLY JSON in a ```json block: decision_request (keep_as_best|reject_and_continue|ready_for_gate|blocked), result_summary, next_direction. Mechanical evidence: "+json.dumps({k:rec.get(k) for k in ("compile","correctness","development_benchmark")},ensure_ascii=False,default=str)
                            advisory=self._reply(self.session.run_experiment_turn(instruction,context))
                            if advisory.get("decision_request") not in {"keep_as_best","reject_and_continue","ready_for_gate","blocked"}:return self._finish("WAITING_FOR_HUMAN",records,reason="INVALID_AGENT_DECISION")
                            rec["agent_interpretation"]=advisory;rec["decision"]="READY_FOR_GATE" if advisory["decision_request"]=="ready_for_gate" else advisory["decision_request"].upper();fails=0
            except (InvalidAgentOutput,PathPolicyViolation) as exc:return self._finish("WAITING_FOR_HUMAN",records,reason=str(exc))
            hyp_record = None
            try:
                hyp = Hypothesis.from_dict((rec.get("plan") or {}).get("hypothesis") or {})
                if hyp.claim:
                    hyp.status = "TESTED"
                    hyp_record = hyp.to_dict()
            except Exception:
                pass
            append_experiment_with_hypothesis(self.journal, rec, hyp_record);self._heartbeat("DECISION", n, experiment_id=experiment_id, candidate_id=rec.get("candidate_id",""));self._emit("AGENT_DECISION",{"experiment_id":rec["experiment_id"],"decision":rec.get("decision"),"experiment_kind":rec.get("experiment_kind")});self._emit("EXPERIMENT_FINISHED",rec);records.append(rec);sync_live(self.live,state="EXPERIMENT_RECORDED",experiment=n,last_record=rec["experiment_id"],environment_fingerprint=self.workbench.get("environment_fingerprint"))
                        # Phase 5-B: reset workspace on rejection
            if rec.get("decision","").startswith("REJECT"):
                self.workspace.reset_to_baseline();self._heartbeat("RESET", n, experiment_id=rec.get("experiment_id",""))
                self._emit("CANDIDATE_RESET",{"reason":rec.get("decision"),"experiment_id":rec.get("experiment_id")})
            if rec.get("decision")=="INVALID_FRAMEWORK_FAILURE":
                detail=(rec.get("framework_failure") or {}).get("reason","unknown framework failure")
                self._emit("EXPERIMENT_RECLASSIFIED",{"experiment_id":rec["experiment_id"],"old_status":"INVALID_FRAMEWORK_FAILURE","new_status":"INVALID_FRAMEWORK_FAILURE","reason":"DIAGNOSTIC_ACTION_DISPATCH_FAILURE","reason_detail":detail,"counts_against_experiment_budget":False,"optimization_evidence":False,"gpu_evidence":False,"logical_experiment_id":logical_id,"attempt":attempt})
                # One automatic retry is allowed for a logical experiment.  A
                # second framework failure is stopped by accounting at the top
                # of this loop and never turns into a GPU result.
                continue
            if rec.get("decision","")=="READY_FOR_GATE":rec["candidate_commit"]=self._commit(rec);return self._gate(rec,records)
            if rec.get("decision","")=="BLOCKED":return self._finish("WAITING_FOR_HUMAN",records,reason="AGENT_BLOCKED")
            if fails>=int(self.budget["max_consecutive_failures"]):return self._finish("WAITING_FOR_HUMAN",records,reason="MAX_CONSECUTIVE_FAILURES")
        finally:self.session.close()
    def _gate(self,selected,records):
        self._status("READY_FOR_GATE",experiment=selected["experiment_id"]);self._emit("HANDOFF_CREATED",{"selected_experiment":selected["experiment_id"],"candidate_commit":selected.get("candidate_commit"),"journal":str(self.journal)});self._emit("SUPERVISOR_STARTED",{"experiment_id":selected["experiment_id"]});self._status("RUNNING_ABBA");abba=self.evaluator.authoritative_abba();decision=decide({"compile":selected["compile"],"correctness":selected["correctness"],"authoritative_abba":abba});decision.experiment_id=selected["experiment_id"];robust=None
        if decision=="ROBUSTNESS_REQUIRED":self._status("RUNNING_ROBUSTNESS");robust=self.evaluator.robustness();decision=finalize_after_robustness(robust);decision.experiment_id=selected["experiment_id"]
        # Save structured decision for audit trail
        decision.save(self.episode_dir / f"decision_{selected['experiment_id']}.json")
        after=str(self.incumbent)
        if decision=="PROMOTE":
            self._status("PROMOTING");self._heartbeat("PROMOTION", selected.get("experiment_id",""), experiment_id=selected.get("experiment_id",""), candidate_id=selected.get("candidate_lineage",{}).get("candidate_id",""));target=self.campaign_dir/"incumbent"/f"{selected['experiment_id']}-{selected['candidate_hash'][:12]}"
            if not target.exists():shutil.copytree(self.candidate_root,target,ignore=shutil.ignore_patterns(".git","__pycache__"))
            after=str(target);self._emit("INCUMBENT_PROMOTED",{"from":str(self.incumbent),"to":after,"candidate_commit":selected.get("candidate_commit")})
        canonical={"schema_version":1,"campaign_id":self.campaign_dir.name,"episode_id":self.episode_dir.name,"timestamp":utc(),"supervisor_decision":decision.action,"supervisor_decision_full":decision.to_dict(),"selected_experiment":selected,"experiments":records,"workbench":{**self.workbench,"working_directory":str(self.candidate_root)},"environment_fingerprint":self.workbench.get("environment_fingerprint"),"authoritative_abba":abba,"robustness":robust,"incumbent_before":str(self.incumbent),"incumbent_after":after,"failures":[r.get("decision","") for r in records if str(r.get("decision","")).startswith("REJECT")],"knowledge_used":selected.get("knowledge",{}),"knowledge_candidates":[],"next_directions":[selected.get("agent_interpretation",{}).get("next_direction",selected["plan"].get("next_direction","UNKNOWN"))],"provenance":{"candidate_hash":selected["candidate_hash"],"candidate_commit":selected.get("candidate_commit"),"journal":str(self.journal)}}
        path=write_canonical(self.campaign_dir/"memory",canonical);candidates=write_knowledge_candidates(self.lab_root,path,canonical);canonical["knowledge_candidates"]=[str(x) for x in candidates];atomic_json(path,canonical);archives=archive_episode(self.lab_root,path,canonical);self._emit("SUPERVISOR_RESULT",{"decision":decision.action,"decision_full":decision.to_dict(),"authoritative_abba":abba,"robustness":robust});self._emit("CANONICAL_MEMORY_WRITTEN",{"path":str(path),"decision":decision.action,"archives":[str(x) for x in archives],"knowledge_candidates":[str(x) for x in candidates]});return self._finish(decision.action,records,canonical_memory=str(path),incumbent_after=after)
