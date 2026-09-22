"""Canonical semantic contracts and fail-closed episode validation.

The legacy V100 campaign used a shape-list digest named ``contract_hash``.
This module keeps that value readable as legacy metadata, but never treats it
as a semantic ABI contract.  New evaluations use a canonical serialized
semantic contract and a separate evaluation fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _shape_list(shapes: list[Any]) -> list[list[int]]:
    normalized: list[list[int]] = []
    for shape in shapes:
        if isinstance(shape, str):
            normalized.append([int(part) for part in shape.split(",")])
        else:
            normalized.append([int(part) for part in shape])
    return normalized


def _argument_marker(arguments: list[dict[str, str]]) -> str:
    return "// AKA_CONTRACT: " + ", ".join(item["name"] for item in arguments)


@dataclass(frozen=True)
class CanonicalContract:
    payload: dict[str, Any]

    @property
    def semantic_contract_sha256(self) -> str:
        return sha256(self.payload)

    # Compatibility helpers for the legacy prompt builder.  The old name is
    # intentionally only an alias inside this in-memory object; artifacts use
    # the unambiguous ``semantic_contract_sha256`` field.
    @property
    def contract_hash(self) -> str:
        return self.semantic_contract_sha256

    @property
    def version(self) -> int:
        return int(self.payload["contract_version"])

    @property
    def entry(self) -> str:
        return self.payload["entry_symbol"]

    @property
    def arguments(self) -> list[dict[str, str]]:
        return self.payload["arguments"]

    @property
    def required_source_marker(self) -> str:
        return self.payload["required_source_marker"]

    def c_signature(self) -> str:
        args = ", ".join(f"{item['type']} {item['name']}" for item in self.arguments)
        return f'extern "C" void {self.entry}({args})'

    def arg_names(self) -> list[str]:
        return [item["name"] for item in self.arguments]

    def contract_marker(self) -> str:
        return self.required_source_marker

    def prompt_roles_section(self) -> str:
        return "\n".join(f"- {item['name']}: {item['role']} ({item['type']})" for item in self.arguments)

    def prompt_semantics_section(self) -> str:
        return self.payload["mathematical_semantics"]


def contract_from_metadata(metadata: dict[str, Any], evaluation: dict[str, Any]) -> CanonicalContract:
    schema = metadata.get("contract_schema") or {}
    arguments = [
        {"name": str(item["name"]), "type": str(item["type"]), "role": str(item["role"])}
        for item in schema.get("arguments", [])
    ]
    if not arguments:
        raise ValueError("operator metadata has no ordered contract arguments")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "operator": str(schema.get("operator") or metadata.get("operator")),
        "contract_version": int(schema.get("version", 1)),
        "interface_type": str(metadata.get("interface", "standalone_cuda")),
        "entry_symbol": str(schema.get("entry", "launch_kernel")),
        "arguments": arguments,
        "dtype": str(metadata.get("dtype", "float32")),
        "semantic_identifier": str(metadata.get("semantic_identifier", "rmsnorm_rowwise_fp32")),
        "mathematical_semantics": str(schema.get("semantics", "")),
        "supported_shapes": _shape_list(evaluation.get("shapes", [])),
        "tolerance": float(evaluation.get("correctness_tolerance", 0.001)),
        "required_source_marker": str(metadata.get("required_source_marker") or _argument_marker(arguments)),
        "target_architecture": str(metadata.get("arch", "")),
    }
    if not payload["operator"] or not payload["mathematical_semantics"]:
        raise ValueError("operator metadata has incomplete semantic contract")
    return CanonicalContract(payload)


def evaluation_fingerprint(evaluation: dict[str, Any]) -> str:
    """Fingerprint evaluation configuration without calling it a contract hash."""
    payload = {
        "shapes": _shape_list(evaluation.get("shapes", [])),
        "score": evaluation.get("score"),
        "warmup": evaluation.get("warmup"),
        "iterations": evaluation.get("iterations"),
        "compiler": evaluation.get("compiler"),
        "compiler_flags": evaluation.get("compiler_flags"),
    }
    return sha256(payload)


def load_contract_bundle(project_root: Path, operator: str) -> tuple[CanonicalContract, dict[str, Any], str]:
    meta_path = project_root / "operators" / operator / "metadata.json"
    eval_path = project_root / "config" / "environments" / "v100_sm70" / f"evaluation_{operator}.json"
    if not eval_path.is_file():
        eval_path = project_root / "config" / "environments" / "v100_sm70" / "evaluation.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    return contract_from_metadata(metadata, evaluation), evaluation, evaluation_fingerprint(evaluation)


def _normalize_type(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("__restrict__", "").strip())


def _source_argument_names_and_types(source: str, entry: str) -> list[tuple[str, str]] | None:
    match = re.search(rf'extern\s+"C"\s+void\s+{re.escape(entry)}\s*\((.*?)\)', source, re.S)
    if not match:
        return None
    parsed: list[tuple[str, str]] = []
    for raw in match.group(1).split(","):
        tokens = raw.strip().split()
        if len(tokens) < 2:
            return None
        name = tokens[-1]
        arg_type = _normalize_type(" ".join(tokens[:-1]))
        parsed.append((name, arg_type))
    return parsed


def validate_candidate_source(candidate_path: Path, contract: CanonicalContract) -> list[dict[str, str]]:
    source = candidate_path.read_text(encoding="utf-8")
    failures: list[dict[str, str]] = []
    if contract.required_source_marker not in source:
        failures.append({"field": "required_source_marker", "expected": contract.required_source_marker, "actual": "missing", "artifact": str(candidate_path)})
    parsed = _source_argument_names_and_types(source, contract.entry)
    if parsed is None:
        failures.append({"field": "interface.entry", "expected": contract.entry, "actual": "missing", "artifact": str(candidate_path)})
        return failures
    expected = [(item["name"], _normalize_type(item["type"])) for item in contract.arguments]
    if parsed != expected:
        failures.append({"field": "interface.ordered_arguments", "expected": repr(expected), "actual": repr(parsed), "artifact": str(candidate_path)})
    return failures


def validate_episode_contract(ep_dir: Path, contract: CanonicalContract, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate all semantic identities before any remote evaluator can run."""
    ep_dir = Path(ep_dir)
    failures = validate_candidate_source(ep_dir / "candidate.cu", contract)
    expected_hash = contract.semantic_contract_sha256
    hypothesis_path = ep_dir / "hypothesis.json"
    hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8")) if hypothesis_path.is_file() else None
    if hypothesis is None:
        failures.append({"field": "hypothesis", "expected": "semantic_contract_sha256", "actual": "missing", "artifact": str(hypothesis_path)})
    else:
        for field, expected in (("operator", contract.payload["operator"]), ("contract_version", contract.payload["contract_version"]), ("semantic_contract_sha256", expected_hash)):
            actual = hypothesis.get(field)
            if actual != expected:
                failures.append({"field": f"hypothesis.{field}", "expected": str(expected), "actual": str(actual) if actual is not None else "missing", "artifact": str(hypothesis_path)})
        interface = hypothesis.get("interface") or {}
        expected_names = [item["name"] for item in contract.arguments]
        if interface.get("entry") != contract.entry:
            failures.append({"field": "hypothesis.interface.entry", "expected": contract.entry, "actual": str(interface.get("entry", "missing")), "artifact": str(hypothesis_path)})
        if interface.get("arguments") != expected_names:
            failures.append({"field": "hypothesis.interface.arguments", "expected": repr(expected_names), "actual": repr(interface.get("arguments")), "artifact": str(hypothesis_path)})
    manifest = manifest if manifest is not None else _read_json(ep_dir / "episode_manifest.json")
    if not manifest:
        failures.append({"field": "manifest", "expected": expected_hash, "actual": "missing", "artifact": str(ep_dir / "episode_manifest.json")})
    else:
        for field, expected in (("operator", contract.payload["operator"]), ("semantic_contract_sha256", expected_hash), ("interface_entry", contract.entry), ("interface_arguments", contract.arguments)):
            actual = manifest.get(field)
            if actual != expected:
                failures.append({"field": f"manifest.{field}", "expected": str(expected), "actual": str(actual) if actual is not None else "missing", "artifact": str(ep_dir / "episode_manifest.json")})
    return {"status": "PASS" if not failures else "REJECT_CONTRACT", "semantic_contract_sha256": expected_hash, "failures": failures}


def inspect_legacy_contract(manifest: dict[str, Any]) -> str:
    if "semantic_contract_sha256" not in manifest:
        return "LEGACY_UNVERIFIED_CONTRACT"
    return "SEMANTIC_CONTRACT_PRESENT"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
