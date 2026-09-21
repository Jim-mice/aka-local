"""Controller-enforced filesystem boundary for an episode candidate."""
from __future__ import annotations
import hashlib
from pathlib import Path


class PathPolicyViolation(RuntimeError):
    pass


class CandidatePathPolicy:
    def __init__(self, candidate_root: Path, protected_roots=()):
        self.candidate_root = Path(candidate_root).resolve()
        self.protected_roots = tuple(Path(path).resolve() for path in protected_roots)

    def allowed(self, path: Path) -> bool:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.candidate_root)
            return not any(resolved.is_relative_to(root) for root in self.protected_roots)
        except ValueError:
            return False

    def require_allowed(self, path: Path) -> Path:
        resolved = Path(path).resolve()
        if not self.allowed(resolved):
            raise PathPolicyViolation(f"PATH_POLICY_VIOLATION:{resolved}")
        return resolved

    @staticmethod
    def snapshot(root: Path) -> dict[str, str]:
        root = Path(root).resolve()
        result = {}
        if not root.exists():
            return result
        for path in root.rglob("*"):
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
                result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    @staticmethod
    def diff(before: dict[str, str], after: dict[str, str]) -> list[str]:
        return sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))

    def validate_snapshot_delta(self, before: dict[str, str], after: dict[str, str]) -> list[str]:
        """Return changed relative paths after enforcing candidate containment.

        Snapshot keys are created from ``candidate_root``.  Resolving every
        changed path here catches a newly-created junction/symlink escape as
        soon as the controller observes it, rather than trusting the model
        prompt alone.
        """
        changed = self.diff(before, after)
        for relative in changed:
            self.require_allowed(self.candidate_root / relative)
        return changed

    def source_line_guard(self) -> dict[str, int]:
        """Capture executable-line counts for existing source files.

        It is intentionally conservative: a candidate may change code, but a
        model cannot turn an existing implementation into an empty/comment-only
        file while claiming a harmless patch.
        """
        result = {}
        for path in self.candidate_root.rglob("*"):
            if path.suffix.lower() not in {".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp"} or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            result[str(path.relative_to(self.candidate_root))] = sum(1 for line in text.splitlines() if line.strip() and not line.lstrip().startswith(("#", "//", "/*", "*")))
        return result

    def validate_source_line_guard(self, before: dict[str, int]) -> None:
        after = self.source_line_guard()
        for relative, count in before.items():
            if after.get(relative, 0) < max(1, count // 2):
                raise PathPolicyViolation(f"PATH_POLICY_VIOLATION: source content was destructively reduced: {relative}")
