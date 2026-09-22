"""Static blueprint model and mock harness for external Megatron integration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from lab.runtime.evaluators.end_to_end_oj import EndToEndOJ, QualificationEvidence
from lab.runtime.evaluators.integration_oj import IntegrationOJ, REQUIRED_CHECKS
from lab.runtime.evaluators.oj_models import PromotionStatus, promotion_for_verdict


def strict_json_loads(payload: str | bytes) -> Any:
    """Decode evidence JSON while rejecting duplicate keys and NaN/Infinity."""

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise TypeError("JSON payload must be text or bytes")
    return json.loads(
        payload,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


@dataclass(frozen=True)
class OperatorBlueprint:
    operator_id: str
    target_commit: str
    megatron_source_point: str
    baseline_implementation: str
    replacement_interface: dict[str, Any]
    real_shape_discovery_method: str
    forward_check: tuple[str, ...]
    backward_gradient_check: tuple[str, ...]
    l0_evaluator: str
    l1_evaluator: str
    l2_evaluator: str
    required_training_command: str
    metrics: tuple[str, ...]
    fallback_detection: str
    artifact_layout: dict[str, str]
    result_schema: str
    promotion_states: tuple[str, ...]
    external_status: str = "EXTERNAL_INTEGRATION_REQUIRED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_blueprint_artifact_manifest(blueprint: OperatorBlueprint, manifest: dict[str, Any]) -> tuple[str, ...]:
    """Validate provenance/layout metadata without touching external artifacts."""
    if not isinstance(manifest, dict):
        return ("manifest_type",)
    failures: list[str] = []
    if manifest.get("operator_id") != blueprint.operator_id:
        failures.append("operator_id")
    if manifest.get("target_commit") != blueprint.target_commit:
        failures.append("target_commit")
    candidate_hash = manifest.get("candidate_hash")
    if not isinstance(candidate_hash, str) or re.fullmatch(r"[0-9a-f]{64}", candidate_hash) is None:
        failures.append("candidate_hash")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        failures.append("artifacts")
    elif set(artifacts) != set(blueprint.artifact_layout):
        failures.append("artifact_roles")
    for role, expected_path in blueprint.artifact_layout.items():
        entry = artifacts.get(role)
        if not isinstance(entry, dict):
            failures.append(f"artifact:{role}")
            continue
        if entry.get("path") != expected_path:
            failures.append(f"artifact_path:{role}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            failures.append(f"artifact_sha256:{role}")
    expected_promotions = set(blueprint.promotion_states)
    promotions = manifest.get("promotion_state")
    if not isinstance(promotions, dict) or set(promotions) != expected_promotions:
        failures.append("promotion_state")
    else:
        allowed = {status.value for status in PromotionStatus}
        if any(value not in allowed for value in promotions.values()):
            failures.append("promotion_state_values")
        elif (
            promotions["IntegrationPromotion"] != PromotionStatus.NOT_EVALUATED.value
            and promotions["KernelPromotion"] != PromotionStatus.ACCEPTED.value
        ) or (
            promotions["SystemPromotion"] != PromotionStatus.NOT_EVALUATED.value
            and promotions["IntegrationPromotion"] != PromotionStatus.ACCEPTED.value
        ):
            failures.append("promotion_state_order")
    if not isinstance(manifest.get("replacement_marker"), str) or not manifest.get("replacement_marker"):
        failures.append("replacement_marker")
    fallback_count = manifest.get("fallback_count")
    if isinstance(fallback_count, bool) or not isinstance(fallback_count, int) or fallback_count != 0:
        failures.append("fallback_count")
    return tuple(failures)


def validate_blueprint_result_bundle(
    blueprint: OperatorBlueprint,
    bundle_root: Path,
    manifest: dict[str, Any],
) -> tuple[str, ...]:
    """Bind declared digests, judge results, and promotion states fail-closed.

    This validator performs no evaluation itself. It verifies an already
    materialized offline bundle and therefore cannot turn invented metrics into
    evidence; the referenced result files must still come from the declared OJ
    runners.
    """
    failures = list(validate_blueprint_artifact_manifest(blueprint, manifest))
    root = Path(bundle_root).resolve()
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, dict):
        return tuple(dict.fromkeys(failures))

    results: dict[str, dict[str, Any]] = {}
    json_artifacts: dict[str, Any] = {}
    for role, entry in artifacts.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        artifact_path = (root / entry["path"]).resolve()
        try:
            artifact_path.relative_to(root)
        except ValueError:
            failures.append(f"artifact_path_escape:{role}")
            continue
        if not artifact_path.is_file():
            failures.append(f"artifact_missing:{role}")
            continue
        try:
            payload = artifact_path.read_bytes()
        except OSError:
            failures.append(f"artifact_unreadable:{role}")
            continue
        if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
            failures.append(f"artifact_digest_mismatch:{role}")
        if role in blueprint.artifact_layout:
            try:
                parsed = strict_json_loads(payload)
            except (UnicodeDecodeError, TypeError, ValueError):
                failures.append(f"artifact_json:{role}")
                continue
            json_artifacts[role] = parsed
        if role in {"l0", "l1", "l2"}:
            if not isinstance(parsed, dict):
                failures.append(f"result_schema:{role}")
                continue
            results[role] = parsed

    contract_artifact = json_artifacts.get("contract")
    contract_commit = contract_artifact.get("target_commit") if isinstance(contract_artifact, dict) else None
    if isinstance(contract_artifact, dict) and contract_commit is None and isinstance(contract_artifact.get("megatron"), dict):
        contract_commit = contract_artifact["megatron"].get("commit")
    if not isinstance(contract_artifact, dict) or contract_artifact.get("operator_id") != blueprint.operator_id or contract_commit != blueprint.target_commit:
        failures.append("contract_artifact_identity")

    shape_artifact = json_artifacts.get("shapes")
    captures = shape_artifact.get("captures") if isinstance(shape_artifact, dict) else None
    if not isinstance(shape_artifact, dict) or shape_artifact.get("operator_id") != blueprint.operator_id or not isinstance(captures, list) or not captures:
        failures.append("shape_manifest")
    elif any(not _valid_shape_capture(capture) for capture in captures):
        failures.append("shape_manifest_capture")

    provenance_artifact = json_artifacts.get("provenance")
    contract_role = artifacts.get("contract")
    contract_digest = contract_role.get("sha256") if isinstance(contract_role, dict) else None
    candidate_hash = manifest.get("candidate_hash")
    if (
        not isinstance(provenance_artifact, dict)
        or provenance_artifact.get("schema_version") != 1
        or provenance_artifact.get("target_commit") != blueprint.target_commit
        or not isinstance(provenance_artifact.get("runner_id"), str)
        or not provenance_artifact.get("runner_id")
        or not isinstance(provenance_artifact.get("environment_hash"), str)
        or re.fullmatch(r"[0-9a-f]{64}", provenance_artifact.get("environment_hash", "")) is None
        or not isinstance(provenance_artifact.get("measurement_protocol"), dict)
        or not provenance_artifact.get("measurement_protocol")
        or provenance_artifact.get("contract_hash") != contract_digest
        or provenance_artifact.get("candidate_hash") != candidate_hash
    ):
        failures.append("provenance_artifact")

    result_contract = {
        "l0": ("L0_OPERATOR", {"OPERATOR_PASS", "OPERATOR_FAIL"}),
        "l1": ("L1_INTEGRATION", {"INTEGRATION_PASS", "INTEGRATION_FAIL"}),
        "l2": ("L2_END_TO_END", {"SYSTEM_PROVISIONAL", "SYSTEM_QUALIFIED_ACCEPT", "SYSTEM_REJECT"}),
    }
    required_fields = {"level", "verdict", "checks", "raw_metrics", "evidence", "reasons"}
    core_checks = {
        "l0": {"compile", "contract", "correctness", "shapes", "stability", "latency_valid", "samples_valid", "contract_hash", "candidate_hash"},
        "l1": {*REQUIRED_CHECKS, "replacement_marker_matches", "fallback_count_zero", "runner_provenance", "contract_hash_matches", "candidate_hash_matches"},
        "l2": {"input_json", "iteration_inputs", "iteration_speedup", "throughput_inputs", "throughput_gain", "samples_per_sec", "peak_memory", "loss_finite", "convergence_proxy_finite", "gradient"},
    }
    for role, (expected_level, allowed_verdicts) in result_contract.items():
        result = results.get(role)
        if result is None:
            continue
        if not required_fields.issubset(result):
            failures.append(f"result_schema:{role}")
        if result.get("level") != expected_level:
            failures.append(f"result_level:{role}")
        if result.get("verdict") not in allowed_verdicts:
            failures.append(f"result_verdict:{role}")
        if not isinstance(result.get("checks"), dict) or not result.get("checks"):
            failures.append(f"result_checks:{role}")
        elif not core_checks[role].issubset(result["checks"]):
            failures.append(f"result_checks_missing:{role}")
        if not isinstance(result.get("raw_metrics"), dict) or not isinstance(result.get("evidence"), dict):
            failures.append(f"result_payload:{role}")
        if not isinstance(result.get("reasons"), list) or any(not isinstance(reason, str) or not reason for reason in result.get("reasons", ())):
            failures.append(f"result_reasons:{role}")

    l2_for_policy = results.get("l2")
    if l2_for_policy is not None:
        l2_evidence = l2_for_policy.get("evidence")
        policy = l2_evidence.get("policy") if isinstance(l2_evidence, dict) else None
        required_metrics = policy.get("required_metrics") if isinstance(policy, dict) else None
        l2_checks = l2_for_policy.get("checks")
        if not isinstance(required_metrics, (list, tuple)) or not required_metrics or any(not isinstance(name, str) or not name for name in required_metrics):
            failures.append("result_policy_metrics:l2")
        elif isinstance(l2_checks, dict) and any(f"metric:{name}" not in l2_checks for name in required_metrics):
            failures.append("result_metric_checks:l2")

    promotions = manifest.get("promotion_state")
    if isinstance(promotions, dict):
        for role, result in results.items():
            try:
                axis, status = promotion_for_verdict(result.get("verdict"))
            except ValueError:
                continue
            if promotions.get(axis) != status.value:
                failures.append(f"promotion_result_mismatch:{role}")

    l0_pass = results.get("l0", {}).get("verdict") == "OPERATOR_PASS"
    l1_pass = results.get("l1", {}).get("verdict") == "INTEGRATION_PASS"
    if "l1" in results and not l0_pass:
        failures.append("result_chain:l1_requires_l0_pass")
    if "l2" in results and not l1_pass:
        failures.append("result_chain:l2_requires_l1_pass")

    l0_evidence = results.get("l0", {}).get("evidence")
    if not isinstance(l0_evidence, dict) or l0_evidence.get("contract_hash") != contract_digest:
        failures.append("contract_identity:l0")
    if not isinstance(l0_evidence, dict) or l0_evidence.get("candidate_hash") != candidate_hash:
        failures.append("candidate_identity:l0")
    l1_evidence = results.get("l1", {}).get("evidence")
    l1_provenance = l1_evidence.get("runner_provenance") if isinstance(l1_evidence, dict) else None
    if not isinstance(l1_provenance, dict) or l1_provenance.get("contract_hash") != contract_digest:
        failures.append("contract_identity:l1")
    if not isinstance(l1_provenance, dict) or l1_provenance.get("candidate_hash") != candidate_hash:
        failures.append("candidate_identity:l1")

    for role, passing_verdict in (("l0", "OPERATOR_PASS"), ("l1", "INTEGRATION_PASS")):
        result = results.get(role)
        if result is not None:
            checks = result.get("checks")
            verdict = result.get("verdict")
            reasons = result.get("reasons")
            if verdict == passing_verdict and (not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values())):
                failures.append(f"passing_result_checks:{role}")
            if verdict != passing_verdict and isinstance(checks, dict) and checks and all(value is True for value in checks.values()):
                failures.append(f"failing_result_checks:{role}")
            if verdict == passing_verdict and reasons:
                failures.append(f"passing_result_reasons:{role}")
            if verdict != passing_verdict and isinstance(reasons, list) and not reasons:
                failures.append(f"failing_result_reasons:{role}")
    l2 = results.get("l2")
    if l2 is not None:
        checks = l2.get("checks")
        evidence = l2.get("evidence")
        verdict = l2.get("verdict")
        reasons = l2.get("reasons")
        check_values = tuple(checks.values()) if isinstance(checks, dict) else ()
        if verdict == "SYSTEM_REJECT" and False not in check_values:
            failures.append("rejected_result_checks:l2")
        if verdict == "SYSTEM_PROVISIONAL" and (False in check_values or None not in check_values):
            failures.append("provisional_result_checks:l2")
        if verdict == "SYSTEM_QUALIFIED_ACCEPT" and reasons:
            failures.append("passing_result_reasons:l2")
        if verdict != "SYSTEM_QUALIFIED_ACCEPT" and isinstance(reasons, list) and not reasons:
            failures.append("nonaccepted_result_reasons:l2")
    if l2 is not None and l2.get("verdict") == "SYSTEM_QUALIFIED_ACCEPT":
        checks = l2.get("checks")
        evidence = l2.get("evidence")
        if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
            failures.append("passing_result_checks:l2")
        qualification_raw = evidence.get("qualification") if isinstance(evidence, dict) else None
        if not isinstance(evidence, dict) or evidence.get("qualified") is not True or not isinstance(qualification_raw, dict):
            failures.append("qualified_result_provenance:l2")
        else:
            policy = evidence.get("policy")
            minimum_runs = policy.get("min_qualification_runs", 2) if isinstance(policy, dict) else 2
            qualification = None
            try:
                qualification = QualificationEvidence.from_dict(qualification_raw)
                qualification_checks = qualification.checks(minimum_runs)
            except (TypeError, ValueError):
                qualification_checks = {}
            if not qualification_checks or not all(qualification_checks.values()):
                failures.append("qualified_result_provenance:l2")
            declared_paths = {
                entry.get("path")
                for entry in artifacts.values()
                if isinstance(entry, dict) and isinstance(entry.get("path"), str)
            }
            if qualification is None or any(reference not in declared_paths for reference in qualification.artifact_refs):
                failures.append("qualification_artifact_binding:l2")
            required_raw_path = blueprint.artifact_layout.get("l2_raw")
            if qualification is None or required_raw_path not in qualification.artifact_refs:
                failures.append("qualification_raw_artifact:l2")
            provenance_role = artifacts.get("provenance")
            if qualification is None or not isinstance(provenance_role, dict) or provenance_role.get("sha256") != qualification.protocol_hash:
                failures.append("qualification_protocol_binding:l2")
            raw_runs = json_artifacts.get("l2_raw")
            if not isinstance(raw_runs, dict):
                failures.append("qualification_run_binding:l2")
            elif qualification is not None:
                def run_ids(name: str) -> tuple[Any, ...]:
                    values = raw_runs.get(name)
                    if not isinstance(values, list):
                        return ()
                    return tuple(item.get("run_id") if isinstance(item, dict) else item for item in values)

                if run_ids("baseline_runs") != qualification.baseline_run_ids or run_ids("candidate_runs") != qualification.candidate_run_ids:
                    failures.append("qualification_run_binding:l2")

    return tuple(dict.fromkeys(failures))


def _valid_shape_capture(capture: Any) -> bool:
    if not isinstance(capture, dict):
        return False
    dimensions = capture.get("dimensions")
    strides = capture.get("strides")
    return (
        isinstance(capture.get("shape_id"), str)
        and bool(capture["shape_id"])
        and isinstance(capture.get("dtype"), str)
        and bool(capture["dtype"])
        and isinstance(dimensions, dict)
        and bool(dimensions)
        and all(isinstance(name, str) and name and isinstance(value, int) and not isinstance(value, bool) and value > 0 for name, value in dimensions.items())
        and isinstance(strides, list)
        and bool(strides)
        and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in strides)
        and isinstance(capture.get("config_hash"), str)
        and re.fullmatch(r"[0-9a-f]{64}", capture["config_hash"]) is not None
    )


class MockBlueprintHarness:
    def __init__(self, integration_runner: Callable[[dict[str, Any]], dict[str, Any]], e2e: EndToEndOJ | None = None):
        self.integration = IntegrationOJ(integration_runner)
        self.e2e = e2e or EndToEndOJ()

    def run_l1(self, request: dict[str, Any]):
        return self.integration.evaluate(request)

    def run_l2(self, metrics: dict[str, Any], *, qualification: QualificationEvidence | dict[str, Any] | None = None):
        return self.e2e.evaluate(metrics, qualification=qualification)


SWIGLU_BLUEPRINT = OperatorBlueprint(
    operator_id="swiglu",
    target_commit="5be9626709af2722333bf54797c954c09edeada3",
    megatron_source_point="megatron/core/transformer/mlp.py::MLP.forward bias_swiglu_impl branch",
    baseline_implementation="Megatron bias_swiglu_impl or declared unfused torch activation selected by config",
    replacement_interface={"input": "FC1 intermediate [S,B,2I/TP]", "bias": "[2I/TP] or null", "output": "[S,B,I/TP]", "dtype": "declared FP16/BF16", "stream": "current PyTorch CUDA stream", "configuration": "use_te_activation_func=False, bias_activation_fusion=True, SiLU gated_linear_unit=True, per_token_scale=None; weighted path is a separate contract"},
    real_shape_discovery_method="capture S,B,H,I,TP,dtype,stride and config flags at MLP.forward before timing; persist without tensor payload",
    forward_check=("compare against SiLU(gate+b0)*(up+b1)", "cover bias and null-bias", "cover captured shapes"),
    backward_gradient_check=("compare dInput and dBias against Megatron/autograd", "check finite gradients", "sample finite differences in FP32 fixture"),
    l0_evaluator="compile, contract, standalone forward/backward correctness, stability, activation latency",
    l1_evaluator="patch declared MLP activation boundary, assert marker and zero fallback, compare full MLP forward/backward and timing",
    l2_evaluator="run declared training command against baseline/candidate with repeated iteration, throughput, memory, loss, and gradient metrics",
    required_training_command="EXTERNAL: exact nine-grid Megatron pretrain command, model config, dataset, world topology, warmup, and measured-step window must be supplied",
    metrics=("activation_latency_ms", "mlp_module_ms", "iteration_ms", "samples_per_sec", "tokens_per_sec", "peak_memory_bytes", "convergence_proxy_delta", "loss_delta", "gradient_check", "fallback_count"),
    fallback_detection="candidate wrapper increments a per-rank marker and hard-rejects unsupported input or TE/weighted dispatch; L1 asserts selected callable identity, expected count, and fallback_count == 0",
    artifact_layout={"contract": "contract.json", "shapes": "shape_manifest.json", "l0": "l0/result.json", "l1": "l1/result.json", "l2": "l2/result.json", "l2_raw": "l2/raw_metrics.json", "provenance": "provenance.json"},
    result_schema="JudgeResult plus independent PromotionState",
    promotion_states=("KernelPromotion", "IntegrationPromotion", "SystemPromotion"),
)


DENSE_ATTENTION_BLUEPRINT = OperatorBlueprint(
    operator_id="dense_fused_attention",
    target_commit="5be9626709af2722333bf54797c954c09edeada3",
    megatron_source_point="megatron/core/transformer/attention.py::Attention._run_core_attention -> DotProductAttention.forward",
    baseline_implementation="declared dense/no-mask/p=0 native DotProductAttention configuration",
    replacement_interface={"qkv": "[S,B,N/TP,Hd]", "output": "[S,B,N/TP*Hd]", "mask": "declared null/no-mask", "dropout": "0 for initial boundary"},
    real_shape_discovery_method="capture sequence lengths, batch, heads-per-TP, head dimension, mask type, dropout, backend, and strides at _run_core_attention",
    forward_check=("FP32 QK-softmax-PV oracle", "Megatron baseline comparison", "shape and stride matrix"),
    backward_gradient_check=("dQ/dK/dV versus baseline", "finite gradients", "saved-softmax compatibility"),
    l0_evaluator="core attention compile/correctness/latency/workspace",
    l1_evaluator="backend replacement marker, no fallback, full attention forward/backward, local timing and memory",
    l2_evaluator="repeated training iteration, throughput, memory, loss, gradient sanity",
    required_training_command="EXTERNAL: exact nine-grid Megatron command and data/topology are required",
    metrics=("core_attention_ms", "attention_module_ms", "iteration_ms", "samples_per_sec", "tokens_per_sec", "peak_memory_bytes", "convergence_proxy_delta", "loss_delta", "gradient_check", "fallback_count"),
    fallback_detection="assert selected core-attention object identity plus per-rank invocation marker and zero fallback count",
    artifact_layout={"contract": "contract.json", "shapes": "shape_manifest.json", "l0": "l0/result.json", "l1": "l1/result.json", "l2": "l2/result.json", "l2_raw": "l2/raw_metrics.json", "provenance": "provenance.json"},
    result_schema="JudgeResult plus independent PromotionState",
    promotion_states=("KernelPromotion", "IntegrationPromotion", "SystemPromotion"),
)
