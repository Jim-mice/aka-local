"""
CUDA environment bootstrap for aka-local on Windows RTX 5060 (sm_120).

Import this module BEFORE any candidate code is loaded.  It guarantees:

- torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
  (nvcc outputs non-ASCII bytes on Chinese Windows; the OEM codec default
   causes UnicodeDecodeError during compiler version detection)

- CUDA architecture flags are consistent
- Console encoding is UTF-8

Why runtime injection is better than editing every candidate:
  1. The Agent (LLM) does not know about this Windows-specific PyTorch bug.
  2. Every candidate author would need to remember the same boilerplate.
  3. The fix applies uniformly to all past and future candidates.
  4. It survives candidate regeneration and does not leak into the
     candidate source that the Agent inspects for context.
"""

from __future__ import annotations

import os
import sys


def _apply() -> None:
    # --- 1. SUBPROCESS_DECODE_ARGS: the critical fix ---
    # torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version()
    # runs nvcc --version and decodes its output with SUBPROCESS_DECODE_ARGS.
    # Default ("oem",) uses the system OEM code page which fails on Chinese
    # Windows when nvcc outputs bytes that are invalid in cp936.
    try:
        import torch.utils.cpp_extension as _cpp_ext
        _cpp_ext.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    except ImportError:
        pass

    # --- 2. Windows console UTF-8 ---
    # Some toolchain subprocesses inherit the console encoding.  Force UTF-8
    # so pipe output is consistently decodable.
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
        # Force the Windows console to use UTF-8 code page (65001).
        try:
            if hasattr(sys, "stdout") and hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys, "stderr") and hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # --- 3. CUDA architecture ---
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("CUDA_ARCH_LIST", "12.0")


# Apply on import so that `import cuda_environment` is sufficient
# even inside a subprocess that has never seen this module before.
_apply()
