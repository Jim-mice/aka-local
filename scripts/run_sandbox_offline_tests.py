"""Run unittest suites with pre-created temp roots in restricted Windows sandboxes.

Some managed Windows sandboxes cannot re-enter directories made by Python's
``tempfile.mkdtemp(mode=0o700)`` because that mode drops the sandbox ACL. This
launcher changes no test assertions: it only allocates directories that the
calling PowerShell process created with inherited ACLs.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from itertools import count
from pathlib import Path


class PrecreatedTemporaryDirectory:
    _counter = count()

    def __init__(self, *args, **kwargs):
        root = os.environ.get("AKA_PRECREATED_TEMP_ROOT")
        if not root:
            raise RuntimeError("AKA_PRECREATED_TEMP_ROOT is required")
        self.name = str(Path(root) / f"case-{next(self._counter):03d}")
        if not Path(self.name).is_dir():
            raise RuntimeError(f"pre-created test directory is missing: {self.name}")

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, traceback):
        shutil.rmtree(self.name)
        return False

    def cleanup(self):
        if Path(self.name).exists():
            shutil.rmtree(self.name)


def _prepare_temp_root() -> tuple[Path, bool]:
    supplied = os.environ.get("AKA_PRECREATED_TEMP_ROOT")
    if supplied:
        root = Path(supplied).resolve()
        if not root.is_dir():
            raise RuntimeError(f"AKA_PRECREATED_TEMP_ROOT is not a directory: {root}")
        return root, False

    project_root = Path(__file__).resolve().parents[1]
    parent = project_root / ".sandbox_offline_tests"
    parent.mkdir(mode=0o777, exist_ok=True)
    root = parent / f"run-{os.getpid()}"
    root.mkdir(mode=0o777)
    os.environ["AKA_PRECREATED_TEMP_ROOT"] = str(root)
    return root, True


def main() -> int:
    root, owned = _prepare_temp_root()
    for index in range(64):
        (root / f"case-{index:03d}").mkdir(mode=0o777, exist_ok=True)
    try:
        tempfile.TemporaryDirectory = PrecreatedTemporaryDirectory
        modules = sys.argv[1:] or ["lab.tests.test_p0_integrity", "lab.tests.test_p1_attempt_loop"]
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    finally:
        if owned:
            shutil.rmtree(root)
            try:
                root.parent.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
