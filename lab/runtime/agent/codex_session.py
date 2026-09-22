"""Persistent, bounded Codex session for one approved episode.

The controller chooses the candidate root.  This adapter never selects a
model, never falls back, and cannot itself decide promotion/evaluation.
"""
import json
from pathlib import Path
from lab.runtime.reasoning.agent_context import augment_authoritative_context, augment_planning_context


class CodexAgentSession:
    model = "gpt-5.6-luna"
    effort = "low"

    def __init__(self, workspace: Path, process_registry=None):
        self.workspace = Path(workspace).resolve()
        self.started = False
        self.stopped = False
        self.client = None
        self.thread = None
        self.last_result = None
        self.context = None
        self.process_registry=process_registry

    def start(self):
        if self.started and not self.stopped:
            return self.status()
        from agent_backends.codex_appserver import CODEX_BIN
        from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
        self._ApprovalMode, self._Sandbox = ApprovalMode, Sandbox
        self.client = Codex(CodexConfig(codex_bin=str(CODEX_BIN), cwd=str(self.workspace), client_name="aka_local_lab", client_title="aka-local GPU Operator Lab", client_version="1.0"))
        self.thread = self.client.thread_start(model=self.model, cwd=str(self.workspace), sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all, ephemeral=True)
        # The SDK owns the app-server Popen.  Attach its actual process instead
        # of guessing from a global codex.exe process list.
        proc=getattr(getattr(self.client,"_client",None),"_proc",None)
        if proc is not None and self.process_registry is not None:self.process_registry.attach(proc,kind="codex_app_server",metadata={"executable":"codex.exe"})
        self.started, self.stopped = True, False
        return self.status()

    @staticmethod
    def prepare_context(context, performance_context=None):
        if performance_context is None:
            return context
        return augment_authoritative_context(context, performance_context)

    def send_context(self, context, performance_context=None):
        context = self.prepare_context(context, performance_context)
        if not isinstance(context,dict) or not context.get("context_hash"):
            raise RuntimeError("CONTEXT_NOT_READY: a hashed filesystem context snapshot is required")
        self.context = context
        return {"status": "CONTEXT_READY", "context_hash": context["context_hash"]}

    def send_planning_context(self, context, planning_context):
        """Attach generated hypotheses while keeping the Agent in planning-only mode."""
        context = augment_planning_context(context, planning_context)
        if not isinstance(context, dict) or not context.get("context_hash"):
            raise RuntimeError("CONTEXT_NOT_READY: a hashed filesystem context snapshot is required")
        self.context = context
        return {"status": "PLANNING_CONTEXT_READY", "context_hash": context["context_hash"], "mode": "PLANNING_ONLY"}

    def run_planning_turn(self, instruction: str, context_snapshot: dict, planning_context: dict):
        """Let the existing session inspect hypotheses; it cannot edit candidates."""
        context_snapshot = augment_planning_context(context_snapshot, planning_context)
        self.send_planning_context(context_snapshot, planning_context)
        if not self.started or self.stopped:
            raise RuntimeError("AgentSession is not active")
        prompt = (
            "PLANNING_ONLY CONTEXT (authoritative; read-only):\n" + self._context_prompt(context_snapshot)
            + "\n\nPLANNING INSTRUCTION:\n" + instruction
            + "\nDo not edit files, create candidates, run benchmarks, or decide promotion."
        )
        self.last_result = self.thread.run(prompt, model=self.model, effort=self.effort, cwd=str(self.workspace), sandbox=self._Sandbox.read_only, approval_mode=self._ApprovalMode.deny_all)
        return self.last_result

    @staticmethod
    def _context_prompt(context):
        return json.dumps(context,ensure_ascii=False,sort_keys=True,default=str)

    def ask_advisory(self, user_question: str, context_snapshot: dict):
        """Answer a human ASK from a supplied Lab snapshot, read-only."""
        self.send_context(context_snapshot)
        if not self.started or self.stopped:
            raise RuntimeError("AgentSession is not active")
        prompt=("You are the read-only advisory assistant for a local GPU Operator Lab.\n"
                "The Lab filesystem/context snapshot below has already been supplied. Do not ask the user to paste Workbench, knowledge, experiment history, or incumbent data that is present in this snapshot. If a requested fact is absent, state exactly which field/evidence is missing. Respond in Simplified Chinese. Preserve technical identifiers, metrics, GPU/backend names, paths, IDs, hashes, shapes, and code symbols exactly. ASK is strictly read-only: do not modify files, run tools, compile, benchmark, profile, or start an experiment.\n\n"
                "CONTEXT SNAPSHOT (authoritative):\n"+self._context_prompt(context_snapshot)+"\n\nUSER QUESTION:\n"+user_question)
        self.last_result=self.thread.run(prompt,model=self.model,effort=self.effort,cwd=str(self.workspace),sandbox=self._Sandbox.read_only,approval_mode=self._ApprovalMode.deny_all)
        return self.last_result

    def run_experiment_turn(self, instruction: str, context_snapshot: dict, performance_context=None):
        """Run a formal turn only when a fresh hashed snapshot is supplied."""
        context_snapshot = self.prepare_context(context_snapshot, performance_context)
        self.send_context(context_snapshot)
        if not self.started or self.stopped:
            raise RuntimeError("AgentSession is not active")
        prompt="FORMAL EXPERIMENT CONTEXT (authoritative; supplied from local filesystem):\n"+self._context_prompt(context_snapshot)+"\n\nEXPERIMENT INSTRUCTION:\n"+instruction
        self.last_result=self.thread.run(prompt,model=self.model,effort=self.effort,cwd=str(self.workspace),sandbox=self._Sandbox.workspace_write,approval_mode=self._ApprovalMode.deny_all)
        return self.last_result

    def ask(self, prompt: str):
        if not self.started or self.stopped:
            raise RuntimeError("AgentSession is not active")
        self.last_result = self.thread.run(prompt, model=self.model, effort=self.effort, cwd=str(self.workspace), sandbox=self._Sandbox.workspace_write, approval_mode=self._ApprovalMode.deny_all)
        return self.last_result

    def send_directive(self, directive):
        return self.ask("Human directive for the next safe experiment: " + str(directive))

    def continue_episode(self, prompt):
        return self.ask(prompt)

    def pause(self):
        return {"status": "PAUSE_REQUESTED"}

    def stop(self):
        self.stopped = True
        return {"status": "STOPPED"}

    def status(self):
        return {"started": self.started, "stopped": self.stopped, "model": self.model, "reasoning_effort": self.effort, "thread_id": getattr(self.thread, "id", None)}

    def close(self):
        self.stop()
        if self.client is not None:
            proc=getattr(getattr(self.client,"_client",None),"_proc",None)
            self.client.close()
            if proc is not None and self.process_registry is not None:self.process_registry.completed(proc.pid,getattr(proc,"returncode",0))
            self.client = self.thread = None
