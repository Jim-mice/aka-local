"""Deterministic supervisor boundary; Agent opinions are advisory only.

Returns structured Decision objects with evidence-backed confidence.
Backward-compatible: ``decision.action`` still works as a string,
and ``decision == "PROMOTE"`` compares against the action field.
"""
from __future__ import annotations

from ...core.evidence import Evidence, EvidenceType, Verdict
from ...core.decision import Decision, compute_confidence


def decide(evaluation, *, weak_gain_pct=2.0):
    """Return a Decision from mechanical evidence only."""
    compile_pass = evaluation.get("compile", {}).get("pass", False)
    correctness_pass = evaluation.get("correctness", {}).get("pass", False)
    abba = evaluation.get("authoritative_abba", {})
    metrics = abba.get("metrics", {})
    speedup = metrics.get("arithmetic_mean_speedup")

    evidence_summary = {
        "compile_pass": compile_pass,
        "correctness_pass": correctness_pass,
        "benchmark_available": speedup is not None,
        "speedup": speedup,
    }

    if not compile_pass:
        return Decision(action="REJECT_COMPILE", reason="Compilation failed",
            confidence=compute_confidence(compile_pass=False),
            evidence_summary=evidence_summary, hypothesis_verdict="REFUTED")
    if not correctness_pass:
        return Decision(action="REJECT_CORRECTNESS", reason="Correctness check failed",
            confidence=compute_confidence(compile_pass=True, correctness_pass=False),
            evidence_summary=evidence_summary, hypothesis_verdict="REFUTED")
    if speedup is None:
        return Decision(action="INCONCLUSIVE", reason="No benchmark data available",
            confidence=compute_confidence(compile_pass=True, correctness_pass=True),
            evidence_summary=evidence_summary, hypothesis_verdict="INCONCLUSIVE")

    improvement_pct = (float(speedup) - 1.0) * 100.0
    evidence_summary["improvement_pct"] = round(improvement_pct, 2)
    evidence_summary["weak_gain_pct"] = weak_gain_pct

    if improvement_pct <= 0:
        return Decision(action="REJECT_PERFORMANCE",
            reason=f"No improvement ({improvement_pct:+.1f}%)",
            confidence=compute_confidence(compile_pass=True, correctness_pass=True, benchmark_available=True, speedup=speedup),
            evidence_summary=evidence_summary, hypothesis_verdict="REFUTED")
    if improvement_pct <= weak_gain_pct:
        return Decision(action="ROBUSTNESS_REQUIRED",
            reason=f"Weak gain ({improvement_pct:+.1f}% <= {weak_gain_pct}%), robustness required",
            confidence=compute_confidence(compile_pass=True, correctness_pass=True, benchmark_available=True, speedup=speedup),
            evidence_summary=evidence_summary, hypothesis_verdict="SUPPORTED")
    return Decision(action="PROMOTE",
        reason=f"Clear improvement ({improvement_pct:+.1f}% > {weak_gain_pct}%)",
        confidence=compute_confidence(compile_pass=True, correctness_pass=True, benchmark_available=True, speedup=speedup),
        evidence_summary=evidence_summary, hypothesis_verdict="SUPPORTED")


def finalize_after_robustness(robustness):
    """Return a Decision from robustness check."""
    robustness_pass = robustness.get("pass", False)
    m = robustness.get("metrics", {}).get("aggregate", {})
    speedup = m.get("arithmetic_mean_speedup")
    evidence = {"robustness_pass": robustness_pass, "robustness_speedup": speedup}
    if not robustness_pass:
        return Decision(action="REJECT_ROBUSTNESS", reason="Robustness check failed",
            confidence=compute_confidence(compile_pass=True, correctness_pass=True, benchmark_available=True, robustness_pass=False),
            evidence_summary=evidence, hypothesis_verdict="REFUTED")
    if speedup is not None and float(speedup) > 1.0:
        return Decision(action="PROMOTE",
            reason=f"Robustness confirmed ({float(speedup):.2f}x)",
            confidence=compute_confidence(compile_pass=True, correctness_pass=True, benchmark_available=True, robustness_pass=True, speedup=speedup),
            evidence_summary=evidence, hypothesis_verdict="SUPPORTED")
    return Decision(action="REJECT_ROBUSTNESS", reason="No improvement in robustness",
        confidence=compute_confidence(compile_pass=True, correctness_pass=True, benchmark_available=True, robustness_pass=True),
        evidence_summary=evidence, hypothesis_verdict="REFUTED")


def decide_from_evidence(evidence=None, *, evaluation=None, weak_gain_pct=2.0):
    """Return a Decision from typed Evidence or raw evaluation dict."""
    if evidence is not None and isinstance(evidence, Evidence):
        evaluation = _evidence_to_evaluation(evidence)
    if evaluation is None:
        return Decision(action="INCONCLUSIVE", reason="No evidence provided")
    return decide(evaluation, weak_gain_pct=weak_gain_pct)


def _evidence_to_evaluation(evidence):
    result = {"compile": {"pass": True}, "correctness": {"pass": True}}
    if evidence.type == EvidenceType.CORRECTNESS:
        if evidence.verdict == Verdict.FAIL:
            result["correctness"] = {"pass": False, "metrics": evidence.metrics}
    elif evidence.type == EvidenceType.PERFORMANCE:
        result["authoritative_abba"] = {"metrics": evidence.metrics, "pass": evidence.verdict == Verdict.PASS}
    elif evidence.type == EvidenceType.PROFILE:
        result["profile"] = {"pass": evidence.verdict == Verdict.PASS, "metrics": evidence.metrics}
    elif evidence.type == EvidenceType.DIAGNOSTIC:
        result["diagnostic"] = {"pass": evidence.verdict == Verdict.PASS, "metrics": evidence.metrics}
    return result
