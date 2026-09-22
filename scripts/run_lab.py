"""Fail-closed launcher for the checkout that contains this file.

It does not select an interpreter.  Callers must select one explicitly; this
launcher only guarantees that every imported ``lab`` module belongs to this
checkout rather than to an editable install or another aka-local tree.
"""
from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = PROJECT_ROOT / "lab"


def _inside_lab_root(value: str | Path) -> bool:
    try:
        Path(value).resolve().relative_to(LAB_ROOT)
        return True
    except (OSError, ValueError):
        return False


def assert_lab_is_local() -> None:
    """Reject preloaded or namespace-merged ``lab`` modules outside this tree."""
    failures: list[str] = []
    for name, module in sorted(sys.modules.items()):
        if name != "lab" and not name.startswith("lab."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and not _inside_lab_root(module_file):
            failures.append(f"{name}.__file__={module_file}")
        module_path = getattr(module, "__path__", None)
        if module_path is not None:
            for entry in module_path:
                if not _inside_lab_root(entry):
                    failures.append(f"{name}.__path__ entry={entry}")
    if failures:
        joined = "\n  ".join(failures)
        raise RuntimeError(
            "REFUSING_TO_RUN: foreign lab module detected.\n"
            f"Expected every lab path below: {LAB_ROOT}\n  {joined}"
        )


def configure_imports() -> None:
    """Put this checkout first and validate before and after importing ``lab``."""
    root_text = str(PROJECT_ROOT)
    assert_lab_is_local()
    sys.path[:] = [root_text] + [entry for entry in sys.path if entry != root_text]
    importlib.import_module("lab")
    assert_lab_is_local()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lab module from this checkout only")
    parser.add_argument("--module", default="lab.cli", help="module to run (default: lab.cli)")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for that module")
    args = parser.parse_args()
    if args.module != "lab.cli" and not args.module.startswith("lab."):
        parser.error("--module must name a lab module")
    configure_imports()
    sys.argv = [args.module, *args.args]
    runpy.run_module(args.module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
