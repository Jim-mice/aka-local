"""Offline, read-only CLI for a materialized operator blueprint bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [str(PROJECT_ROOT)] + [entry for entry in sys.path if entry != str(PROJECT_ROOT)]
from scripts.run_lab import configure_imports

configure_imports()
from lab.runtime.blueprints import DENSE_ATTENTION_BLUEPRINT, SWIGLU_BLUEPRINT, strict_json_loads, validate_blueprint_result_bundle


BLUEPRINTS = {
    SWIGLU_BLUEPRINT.operator_id: SWIGLU_BLUEPRINT,
    DENSE_ATTENTION_BLUEPRINT.operator_id: DENSE_ATTENTION_BLUEPRINT,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=sorted(BLUEPRINTS), required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.bundle_root / "manifest.json"
    try:
        manifest = strict_json_loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"verdict": "BUNDLE_FAIL", "failures": [f"manifest:{exc}"]}, ensure_ascii=False))
        return 1
    if not isinstance(manifest, dict):
        print(json.dumps({"verdict": "BUNDLE_FAIL", "failures": ["manifest_type"]}, ensure_ascii=False))
        return 1
    failures = validate_blueprint_result_bundle(BLUEPRINTS[args.operator], args.bundle_root, manifest)
    print(json.dumps({"verdict": "BUNDLE_PASS" if not failures else "BUNDLE_FAIL", "failures": failures}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
