"""Shared result and promotion types for the three-level judge hierarchy."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class OperatorVerdict(str, Enum):
    OPERATOR_PASS = "OPERATOR_PASS"
    OPERATOR_FAIL = "OPERATOR_FAIL"


class IntegrationVerdict(str, Enum):
    INTEGRATION_PASS = "INTEGRATION_PASS"
    INTEGRATION_FAIL = "INTEGRATION_FAIL"


class SystemVerdict(str, Enum):
    SYSTEM_PROVISIONAL = "SYSTEM_PROVISIONAL"
    SYSTEM_QUALIFIED_ACCEPT = "SYSTEM_QUALIFIED_ACCEPT"
    SYSTEM_REJECT = "SYSTEM_REJECT"


class PromotionStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    REJECTED = "REJECTED"
    PROVISIONAL = "PROVISIONAL"
    ACCEPTED = "ACCEPTED"


VERDICT_PROMOTION: dict[str, tuple[str, PromotionStatus]] = {
    OperatorVerdict.OPERATOR_PASS.value: ("KernelPromotion", PromotionStatus.ACCEPTED),
    OperatorVerdict.OPERATOR_FAIL.value: ("KernelPromotion", PromotionStatus.REJECTED),
    IntegrationVerdict.INTEGRATION_PASS.value: ("IntegrationPromotion", PromotionStatus.ACCEPTED),
    IntegrationVerdict.INTEGRATION_FAIL.value: ("IntegrationPromotion", PromotionStatus.REJECTED),
    SystemVerdict.SYSTEM_PROVISIONAL.value: ("SystemPromotion", PromotionStatus.PROVISIONAL),
    SystemVerdict.SYSTEM_QUALIFIED_ACCEPT.value: ("SystemPromotion", PromotionStatus.ACCEPTED),
    SystemVerdict.SYSTEM_REJECT.value: ("SystemPromotion", PromotionStatus.REJECTED),
}


def promotion_for_verdict(verdict: str) -> tuple[str, PromotionStatus]:
    try:
        return VERDICT_PROMOTION[verdict]
    except KeyError as exc:
        raise ValueError(f"unknown OJ verdict: {verdict}") from exc


def legacy_standalone_promotion(decision: str) -> "PromotionState":
    """Map known legacy standalone decisions to the kernel axis only."""
    mapping = {
        "PROMOTE": PromotionStatus.ACCEPTED,
        "QUALIFIED_ACCEPT": PromotionStatus.ACCEPTED,
        "PROVISIONAL": PromotionStatus.PROVISIONAL,
        "PROVISIONAL_UNSTABLE": PromotionStatus.PROVISIONAL,
        "REJECT": PromotionStatus.REJECTED,
    }
    try:
        status = mapping[decision]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown legacy standalone decision: {decision}") from exc
    state = PromotionState()
    state.set_kernel(status)
    return state


@dataclass(frozen=True)
class JudgeResult:
    level: str
    verdict: str
    checks: dict[str, bool | None]
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionState:
    """Independent promotion axes; lower-level acceptance is never system acceptance."""

    kernel: PromotionStatus = PromotionStatus.NOT_EVALUATED
    integration: PromotionStatus = PromotionStatus.NOT_EVALUATED
    system: PromotionStatus = PromotionStatus.NOT_EVALUATED

    def set_kernel(self, status: PromotionStatus) -> None:
        self.kernel = status
        if status != PromotionStatus.ACCEPTED:
            self.integration = PromotionStatus.NOT_EVALUATED
            self.system = PromotionStatus.NOT_EVALUATED

    def set_integration(self, status: PromotionStatus) -> None:
        if status != PromotionStatus.NOT_EVALUATED and self.kernel != PromotionStatus.ACCEPTED:
            raise ValueError("KernelPromotion must be accepted before IntegrationPromotion is evaluated")
        self.integration = status
        if status != PromotionStatus.ACCEPTED:
            self.system = PromotionStatus.NOT_EVALUATED

    def set_system(self, status: PromotionStatus) -> None:
        if status != PromotionStatus.NOT_EVALUATED and self.integration != PromotionStatus.ACCEPTED:
            raise ValueError("IntegrationPromotion must be accepted before SystemPromotion is evaluated")
        self.system = status

    def apply_result(self, result: JudgeResult) -> None:
        axis, status = promotion_for_verdict(result.verdict)
        expected_level = {
            "KernelPromotion": "L0_OPERATOR",
            "IntegrationPromotion": "L1_INTEGRATION",
            "SystemPromotion": "L2_END_TO_END",
        }[axis]
        if result.level != expected_level:
            raise ValueError(f"verdict {result.verdict} is inconsistent with level {result.level}")
        if axis == "KernelPromotion":
            self.set_kernel(status)
        elif axis == "IntegrationPromotion":
            self.set_integration(status)
        else:
            self.set_system(status)

    def to_dict(self) -> dict[str, str]:
        return {
            "KernelPromotion": self.kernel.value,
            "IntegrationPromotion": self.integration.value,
            "SystemPromotion": self.system.value,
        }
