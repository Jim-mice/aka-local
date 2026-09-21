"""Atrex-style crash recovery: PID guard, heartbeat, and run markers.

RecoveryManager detects interrupted runs and validates whether a
resume is safe.  It never automatically resumes �?the human decides.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..core.persistence import atomic_json, read_json


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


@dataclass
class RecoveryReport:
    """Result of a recovery attempt."""
    run_id: str = ""
    state: str = "CLEAN"  # CLEAN / RUNNING / INTERRUPTED / STALE / BLOCKED
    pid: int = 0
    pid_alive: bool = False
    start_token_valid: bool = False
    heartbeat_stale: bool = False
    last_phase: str = ""
    last_heartbeat: str = ""
    environment_ready: bool = False
    workspace_valid: bool = False
    integrity_pass: bool = False
    can_resume: bool = False
    block_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecoveryManager:
    """Manages crash-safe run markers and heartbeat for one active run.

    Each run gets a unique start_token to guard against PID reuse.
    """

    MARKER_NAME = "recovery_marker.json"
    HEARTBEAT_NAME = "recovery_heartbeat.json"

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        *,
        campaign_dir: Path | None = None,
        candidate_root: Path | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.pid = os.getpid()
        self.start_token = secrets.token_hex(16)
        self.start_time = _utc()
        self.campaign_dir = str(campaign_dir) if campaign_dir else ""
        self.candidate_root = str(candidate_root) if candidate_root else ""

    def _marker_path(self) -> Path:
        return self.run_dir / self.MARKER_NAME

    def _heartbeat_path(self) -> Path:
        return self.run_dir / self.HEARTBEAT_NAME

    def create_marker(self) -> dict:
        """Atomically write the active run marker.

        Must be called at the start of a run, before any Agent or evaluator work.
        """
        payload = {
            "run_id": self.run_id,
            "pid": self.pid,
            "start_token": self.start_token,
            "start_time": self.start_time,
            "campaign_dir": self.campaign_dir,
            "candidate_root": self.candidate_root,
            "state": "RUNNING",
        }
        atomic_json(self._marker_path(), payload)
        return payload

    def update_heartbeat(self, phase: str, experiment: int = 0, *, experiment_id: str = "", candidate_id: str = "") -> dict:
        """Update heartbeat at every key lifecycle stage.

        A stale heartbeat (>60s) means the process is dead.
        """
        payload = {
            "run_id": self.run_id,
            "pid": self.pid,
            "start_token": self.start_token,
            "phase": phase,
            "experiment": experiment,
            "experiment_id": experiment_id,
            "candidate_id": candidate_id,
            "campaign": self.campaign_dir,
            "timestamp": _utc(),
        }
        atomic_json(self._heartbeat_path(), payload)
        return payload

    def clear_marker(self) -> None:
        """Remove the run marker (normal completion)."""
        path = self._marker_path()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def detect(run_dir: Path) -> RecoveryReport:
        """Detect the recovery state of a run directory.

        Returns a RecoveryReport with state: CLEAN / RUNNING / INTERRUPTED / STALE.
        """
        run_dir = Path(run_dir)
        marker_path = run_dir / RecoveryManager.MARKER_NAME
        heartbeat_path = run_dir / RecoveryManager.HEARTBEAT_NAME

        if not marker_path.is_file():
            return RecoveryReport(run_id=run_dir.name, state="CLEAN")

        marker = read_json(marker_path, {}) or {}
        heartbeat = read_json(heartbeat_path, {}) or {}

        report = RecoveryReport(
            run_id=marker.get("run_id", run_dir.name),
            pid=int(marker.get("pid", 0)),
            start_token_valid=True,
            last_phase=heartbeat.get("phase", ""),
            last_heartbeat=heartbeat.get("timestamp", ""),
            details={
                "start_time": marker.get("start_time", ""),
                "campaign_dir": marker.get("campaign_dir", ""),
                "candidate_root": marker.get("candidate_root", ""),
            },
        )

        # Check PID
        if report.pid > 0:
            report.pid_alive = _pid_alive(report.pid)
        else:
            report.pid_alive = False

        # Check start_token (PID reuse guard)
        hb_token = heartbeat.get("start_token", "")
        marker_token = marker.get("start_token", "")
        if hb_token and marker_token and hb_token == marker_token:
            report.start_token_valid = True
        elif report.pid_alive:
            report.start_token_valid = False
        else:
            # PID not alive, token doesn't matter
            report.start_token_valid = True

        # Check heartbeat staleness
        if report.last_heartbeat:
            try:
                hb_time = datetime.fromisoformat(report.last_heartbeat)
                now_time = datetime.now(timezone.utc)
                age = (now_time - hb_time).total_seconds()
                report.heartbeat_stale = age > 60  # 60s staleness threshold
            except (ValueError, TypeError):
                report.heartbeat_stale = True
        else:
            report.heartbeat_stale = True

        # Determine state
        if report.pid_alive and report.start_token_valid and not report.heartbeat_stale:
            report.state = "RUNNING"
            report.can_resume = False
            report.block_reason = "Run is still active"
        elif not report.pid_alive:
            report.state = "INTERRUPTED"
            report.can_resume = True
        elif report.heartbeat_stale and not report.pid_alive:
            report.state = "STALE"
            report.can_resume = True
        else:
            report.state = "INTERRUPTED"
            report.can_resume = True

        # If PID is alive but token mismatch �?possible PID reuse
        if report.pid_alive and not report.start_token_valid:
            report.state = "STALE"
            report.can_resume = True
            report.block_reason = "PID reused by different process"

        return report

    @staticmethod
    def recover(
        run_dir: Path,
        *,
        probe_fn: Callable[[], dict] | None = None,
        workspace_verify_fn: Callable[[], dict] | None = None,
        event_callback: Callable[[str, dict], None] | None = None,
    ) -> RecoveryReport:
        """Validate environment and workspace integrity for a safe resume.

        1. detect()
        2. PID check
        3. GPU re-probe
        4. Workspace integrity check
        5. Return RecoveryReport for human decision.

        Does NOT auto-resume.
        """
        report = RecoveryManager.detect(run_dir)

        if not report.can_resume:
            return report

        # Environment re-probe
        if probe_fn is not None:
            try:
                probe = probe_fn()
                report.environment_ready = probe.get("status") == "READY"
                report.details["probe"] = probe
            except Exception as exc:
                report.environment_ready = False
                report.details["probe_error"] = str(exc)
        else:
            report.environment_ready = True

        # Workspace existence check
        candidate_root = report.details.get("candidate_root", "")
        if candidate_root and Path(candidate_root).is_dir():
            report.workspace_valid = True
        elif not candidate_root:
            report.workspace_valid = True
        else:
            report.workspace_valid = False

        # Snapshot hash verification (built-in, before callback)
        if report.workspace_valid and candidate_root:
            snapshot_path = run_dir / "workspace_snapshot.json"
            if snapshot_path.is_file():
                try:
                    import json as _json
                    snap_data = _json.loads(snapshot_path.read_text(encoding="utf-8"))
                    snap_hash = snap_data.get("candidate_hash", "")
                    if snap_hash:
                        from .workspace import _file_digest
                        current_hash = _file_digest(Path(candidate_root))
                        if current_hash != snap_hash:
                            report.integrity_pass = False
                            if event_callback:
                                event_callback("WORKSPACE_INTEGRITY_CHECKED", {"pass": False, "snapshot_hash": snap_hash, "current_hash": current_hash, "detail": "Workspace files have been modified since snapshot"})
                            report.workspace_valid = False
                            report.can_resume = False
                            report.state = "BLOCKED"
                            report.block_reason = "RECOVERY_BLOCKED: workspace modified"
                            report.details["integrity"] = {
                                "pass": False,
                                "snapshot_hash": snap_hash,
                                "current_hash": current_hash,
                                "detail": "Workspace files have been modified since snapshot",
                            }
                            return report
                except Exception:
                    pass

        # Workspace integrity check (callback-based)
        if report.workspace_valid and workspace_verify_fn is not None:
            try:
                integrity = workspace_verify_fn()
                report.integrity_pass = integrity.get("pass", True)
                report.details["integrity"] = integrity
                if not report.integrity_pass:
                    report.workspace_valid = False
                    report.details["modified_files"] = integrity.get("modified_files", [])
                if event_callback:
                    event_callback("WORKSPACE_INTEGRITY_CHECKED", integrity)
            except Exception as exc:
                report.details["integrity_error"] = str(exc)

        # Final verdict
        if not report.environment_ready:
            report.can_resume = False
            report.state = "BLOCKED"
            report.block_reason = report.details.get("probe_error") or "Environment not ready"
        elif not report.workspace_valid:
            report.can_resume = False
            report.state = "BLOCKED"
            report.block_reason = report.block_reason or "Workspace integrity check failed"
        else:
            report.state = "INTERRUPTED"
            report.can_resume = True

        if event_callback:
            event_callback("RECOVERY_INSPECTED", report.to_dict())
        return report
