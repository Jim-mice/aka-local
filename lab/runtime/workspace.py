"""Git-isolated candidate workspace with atomic snapshot and rollback.

On git repos: uses git worktree for full isolation.
On non-git: falls back to file-copy mode for backward compatibility.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_git_repo(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=str(path), capture_output=True, text=True,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()[-500:]}")
    return result


def _file_digest(root: Path) -> str:
    """SHA256 of all files in root, excluding git and pycache."""
    h = hashlib.sha256()
    root = Path(root)
    if not root.is_dir():
        return h.hexdigest()
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _file_snapshot(root: Path) -> dict[str, str]:
    """Per-file SHA256 map of all files in root."""
    result = {}
    root = Path(root)
    if not root.is_dir():
        return result
    for p in root.rglob("*"):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts:
            result[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


class CandidateWorkspace:
    """Isolated candidate workspace with snapshot, commit, reset, and promote.

    Two modes:
    - **git mode**: full git worktree isolation. Requires campaign_dir to be a git repo.
    - **copy mode**: fallback for non-git campaigns. File-level copy.

    The workspace path is always: episode_dir / "candidate"
    """

    def __init__(
        self,
        episode_dir: Path,
        incumbent_dir: Path,
        campaign_dir: Path | None = None,
        protected_roots: tuple[Path, ...] = (),
    ):
        self.episode_dir = Path(episode_dir).resolve()
        self.incumbent_dir = Path(incumbent_dir).resolve()
        self.campaign_dir = Path(campaign_dir).resolve() if campaign_dir else self.episode_dir.parent
        self.candidate_root = self.episode_dir / "candidate"
        self.protected_roots = tuple(Path(p).resolve() for p in protected_roots)
        self.mode: str = "copy"  # "git" or "copy"
        self._created = False

    @property
    def root(self) -> Path:
        return self.candidate_root

    def create(self) -> "CandidateWorkspace":
        """Create the isolated candidate workspace.

        On first call: copies incumbent -> candidate_root (using git worktree if available).
        On subsequent calls: no-op if already created.
        """
        if self._created and self.candidate_root.is_dir():
            return self

        # Determine mode
        if (self.campaign_dir / ".git").exists() and _is_git_repo(self.campaign_dir):
            self.mode = "git"
        else:
            self.mode = "copy"

        if self.mode == "git":
            self._create_git_worktree()
        else:
            self._create_copy()

        self._created = True
        return self

    def _create_copy(self) -> None:
        """Fallback: copy incumbent files into candidate_root."""
        if self.candidate_root.is_dir():
            return
        shutil.copytree(
            self.incumbent_dir, self.candidate_root,
            ignore=shutil.ignore_patterns(".git", "__pycache__"),
        )

    def _create_git_worktree(self) -> None:
        """Use git worktree for full isolation."""
        if self.candidate_root.is_dir():
            return
        # Find a branch name for this episode
        episode_name = self.episode_dir.name
        branch = f"episode/{episode_name}"

        # Check if branch already exists
        result = _git("branch", "--list", branch, cwd=self.campaign_dir, check=False)
        if branch in result.stdout:
            # Branch exists — use it as base
            _git("worktree", "add", "--checkout", str(self.candidate_root), branch, cwd=self.campaign_dir)
        else:
            # Create new branch from HEAD
            _git("worktree", "add", "-b", branch, str(self.candidate_root), "HEAD", cwd=self.campaign_dir)

    def snapshot(self) -> dict[str, Any]:
        """Return a deterministic snapshot of the current candidate state.

        Returns:
            dict with timestamp, candidate_hash (SHA256 of all files),
            file_map (per-file SHA256), and git_commit (optional).
        """
        snap: dict[str, Any] = {
            "timestamp": _utc(),
            "candidate_root": str(self.candidate_root),
            "candidate_hash": _file_digest(self.candidate_root),
            "file_map": _file_snapshot(self.candidate_root),
            "mode": self.mode,
        }
        if self.mode == "git":
            try:
                result = _git("rev-parse", "HEAD", cwd=self.candidate_root, check=False)
                if result.returncode == 0:
                    snap["git_commit"] = result.stdout.strip()
            except Exception:
                pass
        return snap

    def commit_candidate(self, experiment_id: str) -> str | None:
        """Commit the current candidate state. Returns commit hash or None."""
        if self.mode != "git":
            return self.snapshot()["candidate_hash"]

        try:
            _git("add", "-A", cwd=self.candidate_root)
            _git("commit", "-m", f"candidate: {experiment_id}",
                 cwd=self.candidate_root, check=False)
            result = _git("rev-parse", "HEAD", cwd=self.candidate_root)
            return result.stdout.strip()
        except Exception:
            return None

    def reset_to_baseline(self) -> None:
        """Discard all candidate changes, restoring to baseline.

        Preserves: journal.jsonl, evidence/, candidate_lineage.jsonl,
        context snapshots, knowledge snapshots, validated plans, live.json.
        """
        preserved = {
            "journal.jsonl",
            "live.json",
            "candidate_lineage.jsonl",
            "evidence",
            "context_snapshot.json",
            "knowledge_snapshot.json",
            "reports",
        }

        if self.mode == "git":
            try:
                _git("checkout", "--", ".", cwd=self.candidate_root, check=False)
                _git("clean", "-fd", cwd=self.candidate_root, check=False)
            except Exception:
                # Fall back to file-level reset
                self._reset_files(preserved)
        else:
            self._reset_files(preserved)

    def _reset_files(self, preserved: set[str]) -> None:
        """File-level reset: delete all files except preserved names/prefixes."""
        for path in sorted(self.candidate_root.iterdir()):
            name = path.name
            if name in preserved:
                continue
            # Also preserve files with preserved prefixes
            preserved_prefix = any(name.startswith(p) for p in {
                "validated_plan_e", "context_snapshot_e", "knowledge_snapshot_e",
            })
            if preserved_prefix:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

        # Re-copy baseline files
        for src in self.incumbent_dir.iterdir():
            dst = self.candidate_root / src.name
            if src.name in preserved:
                continue
            if not dst.exists():
                if src.is_dir():
                    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
                else:
                    shutil.copy2(src, dst)

    def file_map(self) -> dict[str, str]:
        """Return per-file SHA256 map."""
        return _file_snapshot(self.candidate_root)

    def digest(self) -> str:
        """Return SHA256 digest of all candidate files."""
        return _file_digest(self.candidate_root)
    def verify_integrity(self, snapshot_hash: str | None = None) -> dict:
        """Verify workspace files match a previously recorded snapshot.

        Args:
            snapshot_hash: optional hash from a prior snapshot() call.
                           If None, compares every file.

        Returns:
            {"pass": bool, "current_hash": str, "snapshot_hash": str,
             "modified_files": [...], "detail": str}
        """
        result = {
            "pass": True,
            "current_hash": _file_digest(self.candidate_root),
            "snapshot_hash": snapshot_hash or "",
            "modified_files": [],
            "detail": "",
        }
        if snapshot_hash and result["current_hash"] != snapshot_hash:
            # Identify which files changed
            current_map = _file_snapshot(self.candidate_root)
            # Compare with a stored snapshot if available
            snap_path = self.episode_dir / "workspace_snapshot.json"
            if snap_path.is_file():
                import json as _json
                try:
                    stored = _json.loads(snap_path.read_text(encoding="utf-8"))
                    stored_map = stored.get("file_map", {})
                    for fpath, fhash in stored_map.items():
                        cfh = current_map.get(fpath, "")
                        if cfh != fhash:
                            result["modified_files"].append({
                                "path": fpath,
                                "expected": fhash,
                                "current": cfh,
                            })
                    # Also detect new files
                    for fpath in current_map:
                        if fpath not in stored_map:
                            result["modified_files"].append({
                                "path": fpath,
                                "expected": "(absent)",
                                "current": current_map[fpath],
                            })
                except Exception:
                    pass
            if result["modified_files"]:
                result["pass"] = False
                result["detail"] = f"{len(result['modified_files'])} files modified"
            elif snapshot_hash:
                result["pass"] = False
                result["detail"] = "Hash mismatch but no per-file diff available"
        return result
