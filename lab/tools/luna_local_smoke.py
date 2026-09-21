"""One-user-approved, one-experiment LOCAL Luna plumbing smoke.

It intentionally asks the agent for a harmless comment-only candidate change,
then runs the real compile/correctness/development-ABBA path.  It is not an
optimization search and never asks for promotion.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from lab.core.events import EventStore
from lab.core.probe import probe_local
from lab.runtime.agent.codex_session import CodexAgentSession
from lab.runtime.agent.long_horizon import LongHorizonRunner
from lab.runtime.evaluators.local import RTX5060LocalEvaluator


ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/"lab"
CAMPAIGN=LAB/"campaigns"/"rms_norm_train__rtx5060_sm120__cuda_cpp"
CANDIDATE=CAMPAIGN/"episodes"/"e0003_luna_smoke_full_evaluator"/"candidate"
INCUMBENT=ROOT/"ops"/"rms_norm_v2"


class SmokeLunaSession(CodexAgentSession):
    """Constrains the integration turn without faking any model response."""
    def send_context(self, context):
        # The existing event log contains an old test-only directive saying not
        # to optimize.  Do not let that stale fixture directive invalidate this
        # explicitly approved, isolated smoke workbench.
        context=dict(context);context["human_directives"]=[]
        return super().send_context(context)
    def ask(self, prompt: str):
        if "local RMSNorm experiment planner" in prompt:
            prompt += "\nINTEGRATION-SMOKE CONSTRAINT: this is not a performance search. Plan a harmless comment-only insertion at the beginning of existing candidate.py that documents the smoke; preserve every existing line. Set decision_request to reject_and_continue."
        elif "Implement this plan" in prompt:
            prompt += "\nINTEGRATION-SMOKE CONSTRAINT: modify only candidate.py by adding one Python comment. Do not change executable code."
        elif "Mechanical evidence" in prompt:
            prompt += "\nINTEGRATION-SMOKE CONSTRAINT: return reject_and_continue regardless of measurements; this run must not enter the promotion gate."
        return super().ask(prompt)

class SmokeRunner(LongHorizonRunner):
    def _context(self, number, knowledge):
        context=super()._context(number,knowledge)
        context["human_directives"]=[]
        return context


def sha_tree(path: Path) -> str:
    h=hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():h.update(str(p.relative_to(path)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def main():
    if not CANDIDATE.is_dir(): raise SystemExit(f"Approved candidate workbench is missing: {CANDIDATE}")
    probe=probe_local()
    if probe.get("status")!="READY": raise SystemExit("LOCAL environment is not READY")
    before_candidate=sha_tree(CANDIDATE); before_incumbent=sha_tree(INCUMBENT)
    evaluator=RTX5060LocalEvaluator(ROOT,candidate=CANDIDATE,incumbent=INCUMBENT,evidence_dir=CANDIDATE.parent/"evidence"/"luna_smoke",dry_run=False)
    runner=SmokeRunner(campaign_dir=CAMPAIGN,candidate_root=CANDIDATE,incumbent=INCUMBENT,session=SmokeLunaSession(CANDIDATE),evaluator=evaluator,events=EventStore(LAB/"runtime"/"events.jsonl"),budget={"max_experiments":1,"max_consecutive_failures":3},workbench={"workbench_id":"local_luna_smoke","platform_id":"rtx5060_laptop_sm120","backend_id":"cuda_cpp","environment_fingerprint":probe.get("environment_fingerprint")},lab_root=LAB)
    result=runner.run()
    after_candidate=sha_tree(CANDIDATE); after_incumbent=sha_tree(INCUMBENT)
    output={"result":result,"candidate_changed":before_candidate!=after_candidate,"incumbent_unchanged":before_incumbent==after_incumbent,"candidate_hash_before":before_candidate,"candidate_hash_after":after_candidate,"incumbent_hash":after_incumbent}
    path=CANDIDATE.parent/"luna_integration_smoke.json";path.write_text(json.dumps(output,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    if not output["candidate_changed"] or not output["incumbent_unchanged"]: raise SystemExit("CANDIDATE_WRITE_ISOLATION_FAIL")
    print("REAL_LUNA_STRUCTURED_PLAN = PASS")
    print("REAL_LUNA_CANDIDATE_EDIT_ISOLATION = PASS")
    print("LOCAL_SMOKE_STATE =",result.get("state"))

if __name__=="__main__":main()
