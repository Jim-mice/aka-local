"""Fail-closed validation for pinned Megatron operator integration contracts."""
from __future__ import annotations

from pathlib import PurePosixPath
import re
from typing import Any


def validate_integration_contract(
    schema: dict[str, Any],
    contract: dict[str, Any],
    *,
    target_commit: str,
) -> tuple[str, ...]:
    failures: list[str] = []
    if not isinstance(schema, dict) or not isinstance(contract, dict):
        return ("document_type",)
    if contract.get("schema_version") != schema.get("schema_version"):
        failures.append("schema_version")

    required = schema.get("required")
    if not isinstance(required, list) or any(not isinstance(name, str) or not name for name in required):
        return tuple(dict.fromkeys([*failures, "schema_required"]))
    missing = sorted(set(required) - set(contract))
    failures.extend(f"missing:{name}" for name in missing)
    unknown = sorted(set(contract) - set(required) - {"schema_version"})
    failures.extend(f"unknown:{name}" for name in unknown)

    sections = schema.get("section_required")
    if not isinstance(sections, dict):
        return tuple(dict.fromkeys([*failures, "schema_sections"]))
    for section, fields in sections.items():
        payload = contract.get(section)
        if not isinstance(payload, dict):
            failures.append(f"section:{section}")
            continue
        if not isinstance(fields, list):
            failures.append(f"schema_section:{section}")
            continue
        failures.extend(f"missing:{section}.{field}" for field in fields if field not in payload)

    operator_id = contract.get("operator_id")
    if not isinstance(operator_id, str) or re.fullmatch(r"[a-z][a-z0-9_]*", operator_id) is None:
        failures.append("operator_id")
    if not isinstance(contract.get("human_name"), str) or not contract["human_name"].strip():
        failures.append("human_name")

    megatron = contract.get("megatron")
    if isinstance(megatron, dict):
        commit = megatron.get("commit")
        if commit != target_commit or re.fullmatch(r"[0-9a-f]{40}", str(commit)) is None:
            failures.append("megatron.commit")
        source_files = megatron.get("source_files")
        if not _nonempty_strings(source_files):
            failures.append("megatron.source_files")
        else:
            for source in source_files:
                path = PurePosixPath(source)
                if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
                    failures.append("megatron.source_files")
                    break
        for name in ("classes", "functions"):
            if not _nonempty_strings(megatron.get(name)):
                failures.append(f"megatron.{name}")
        for name in ("forward_callsite", "backward_relevance", "module_path", "graph_pattern"):
            if not _nonempty_string(megatron.get(name)):
                failures.append(f"megatron.{name}")

    tensor = contract.get("tensor_contract")
    if isinstance(tensor, dict):
        for name in ("inputs", "outputs", "dtype"):
            if not _nonempty_strings(tensor.get(name)):
                failures.append(f"tensor_contract.{name}")
        if not isinstance(tensor.get("symbolic_shapes"), dict) or not tensor["symbolic_shapes"]:
            failures.append("tensor_contract.symbolic_shapes")
        for name in ("layout", "training_inference_distinction"):
            if not _nonempty_string(tensor.get(name)):
                failures.append(f"tensor_contract.{name}")

    for section, names in {
        "distributed_context": ("tensor_parallel", "sequence_parallel", "expert_parallel", "data_parallel"),
        "replacement_boundary": ("replace", "fallback_detection"),
    }.items():
        payload = contract.get(section)
        if isinstance(payload, dict):
            for name in names:
                if not _nonempty_string(payload.get(name)):
                    failures.append(f"{section}.{name}")
    replacement = contract.get("replacement_boundary")
    if isinstance(replacement, dict) and not _nonempty_strings(replacement.get("must_remain_untouched")):
        failures.append("replacement_boundary.must_remain_untouched")

    for section in ("correctness", "performance"):
        payload = contract.get(section)
        fields = sections.get(section, ())
        if isinstance(payload, dict) and isinstance(fields, list):
            for name in fields:
                if not _nonempty_strings(payload.get(name)):
                    failures.append(f"{section}.{name}")
    return tuple(dict.fromkeys(failures))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_strings(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_nonempty_string(item) for item in value)
