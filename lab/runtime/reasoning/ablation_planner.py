"""Minimal one-factor and interaction plan for multi-transformation candidates."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationVariant:
    variant_id: str
    enabled: tuple[str, ...]
    purpose: str


class ExperimentPlanner:
    def plan(self, transformations: list[str] | tuple[str, ...]) -> list[AblationVariant]:
        if not isinstance(transformations, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in transformations):
            raise ValueError("transformations must be a list or tuple of non-empty names")
        ordered = tuple(dict.fromkeys(transformations))
        if not ordered:
            return []
        variants = [AblationVariant("baseline", (), "measure unchanged baseline under the same protocol")]
        variants.extend(AblationVariant(f"only_{name}", (name,), f"isolate the main effect of {name}") for name in ordered)
        if len(ordered) > 1:
            variants.extend(
                AblationVariant(f"without_{name}", tuple(item for item in ordered if item != name), f"measure marginal contribution of {name} in the combined candidate")
                for name in ordered
            )
        variants.append(AblationVariant("combined", ordered, "reproduce the complete candidate"))
        deduplicated: dict[tuple[str, ...], AblationVariant] = {}
        for variant in variants:
            deduplicated.setdefault(variant.enabled, variant)
        return list(deduplicated.values())
