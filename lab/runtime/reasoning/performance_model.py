"""Mechanism-level representation used before candidate implementation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import isfinite
from typing import Any


class Magnitude(str, Enum):
    LARGE = "LARGE"
    MEDIUM = "MEDIUM"
    SMALL = "SMALL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PerformanceFacts:
    operator: str
    shape: dict[str, int | str]
    dtype: str
    dataflow: tuple[str, ...] = ()
    tensor_lifetimes: dict[str, str] = field(default_factory=dict)
    global_memory_reads: tuple[str, ...] = ()
    global_memory_writes: tuple[str, ...] = ()
    shared_residency: tuple[str, ...] = ()
    register_residency: tuple[str, ...] = ()
    estimated_bytes: int | None = None
    estimated_flops: int | None = None
    reductions: tuple[str, ...] = ()
    synchronizations: tuple[str, ...] = ()
    kernel_launches: int | None = None
    producer_consumer_boundaries: tuple[str, ...] = ()
    known_reuse: tuple[str, ...] = ()
    fixed_dimensions: dict[str, int] = field(default_factory=dict)
    dynamic_dimensions: tuple[str, ...] = ()
    parallel_mapping: dict[str, str] = field(default_factory=dict)
    profile_evidence: dict[str, Any] = field(default_factory=dict)
    unknown_fields: tuple[str, ...] = ()
    e2e_profile: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operator, str) or not self.operator or not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("operator and dtype are required")
        if not isinstance(self.shape, dict) or not self.shape or any(
            not isinstance(name, str) or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, str))
            or (isinstance(value, int) and value <= 0)
            or (isinstance(value, str) and not value)
            for name, value in self.shape.items()
        ):
            raise ValueError("shape must map names to positive integer or symbolic dimensions")
        for name in ("estimated_bytes", "estimated_flops", "kernel_launches"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if not isinstance(self.fixed_dimensions, dict) or any(
            not isinstance(name, str) or not name or isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for name, value in self.fixed_dimensions.items()
        ):
            raise ValueError("fixed_dimensions must map names to positive integers")
        tuple_fields = (
            "dataflow", "global_memory_reads", "global_memory_writes",
            "shared_residency", "register_residency", "reductions",
            "synchronizations", "producer_consumer_boundaries", "known_reuse",
            "dynamic_dimensions", "unknown_fields",
        )
        for name in tuple_fields:
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(item, str) or not item for item in value):
                raise ValueError(f"{name} must be a tuple of non-empty strings")
        for name in ("tensor_lifetimes", "parallel_mapping"):
            value = getattr(self, name)
            if not isinstance(value, dict) or any(
                not isinstance(key, str) or not key
                or not isinstance(item, str) or not item
                for key, item in value.items()
            ):
                raise ValueError(f"{name} must map non-empty strings to non-empty strings")
        for name in ("profile_evidence", "e2e_profile"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be a dictionary")
        for name in ("idle_resources", "supports_mechanism_ids", "contradicts_mechanism_ids"):
            value = self.profile_evidence.get(name, ())
            if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item for item in value):
                raise ValueError(f"profile_evidence.{name} must be a list or tuple of non-empty strings")
        if "working_set_feasible" in self.profile_evidence and not isinstance(self.profile_evidence["working_set_feasible"], bool):
            raise ValueError("profile_evidence.working_set_feasible must be boolean")
        if "estimated_bytes_saved" in self.profile_evidence:
            value = self.profile_evidence["estimated_bytes_saved"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or value < 0:
                raise ValueError("profile_evidence.estimated_bytes_saved must be finite and nonnegative")
        scoped = self.profile_evidence.get("mechanism_estimates")
        if scoped is not None:
            if not isinstance(scoped, dict) or any(not isinstance(key, str) or not key or not isinstance(value, dict) for key, value in scoped.items()):
                raise ValueError("profile_evidence.mechanism_estimates must map mechanism IDs to dictionaries")
            for mechanism_id, estimate in scoped.items():
                feasibility = estimate.get("working_set_feasible")
                if feasibility is not None and not isinstance(feasibility, bool):
                    raise ValueError(f"mechanism_estimates.{mechanism_id}.working_set_feasible must be boolean")
                bytes_saved = estimate.get("estimated_bytes_saved")
                if bytes_saved is not None and (isinstance(bytes_saved, bool) or not isinstance(bytes_saved, (int, float)) or not isfinite(float(bytes_saved)) or bytes_saved < 0):
                    raise ValueError(f"mechanism_estimates.{mechanism_id}.estimated_bytes_saved must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Opportunity:
    observation: str
    mechanism: str
    transformation: str
    preconditions: tuple[str, ...]
    expected_effect: tuple[str, ...]
    risks: tuple[str, ...]
    estimated_magnitude: Magnitude
    confidence: str
    required_evidence: tuple[str, ...]
    score: float = 0.0
    mechanism_id: str | None = None
    ranking_factors: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["estimated_magnitude"] = self.estimated_magnitude.value
        return result
