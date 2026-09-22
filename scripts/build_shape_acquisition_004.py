from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP3 = ROOT / "artifacts" / "integration" / "swiglu" / "real_loop_003"
OUT = ROOT / "artifacts" / "integration" / "swiglu" / "real_loop_004"
MEGATRON = Path(r"C:\Users\38154\projects\megatron-lm")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_ref(path: Path, evidence_type: str, notes: str) -> dict[str, object]:
    return {"source": str(path), "source_hash": sha256(path) if path.exists() else None, "evidence_type": evidence_type, "notes": notes}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_json(LOOP3 / "batched_swiglu_timing.json")
    timing = read_json(LOOP3 / "timing_attribution.json")
    authoritative = raw["per_invocation"]["median"]
    reconciliation = {
        "status": "PASS",
        "authoritative_value_us": authoritative,
        "source_artifact": str(LOOP3 / "batched_swiglu_timing.json"),
        "source_artifact_sha256": sha256(LOOP3 / "batched_swiglu_timing.json"),
        "sample_protocol": "outer CUDA Event around authentic bias_swiglu_impl; no per-call inner Event",
        "K": raw["k"],
        "warmup_batches": timing["warmup_batches"],
        "measured_batches": timing["measured_batches"],
        "measured_window_median_ms": raw["window_median_ms"],
        "other_value_us": 55.63945323228836,
        "other_value_origin": "Earlier K=8192 diagnostic batch preserved only in derived measurement_facts_003.json and the previous report final block; its raw batch artifact was overwritten by the final K=512 rerun.",
        "selection_rule": "latest complete raw artifact using the final compliant 20-50 ms window protocol; no benchmark rerun in reconciliation",
    }
    (OUT / "timing_evidence_reconciliation.json").write_text(json.dumps(reconciliation, indent=2), encoding="utf-8")

    # Repair only derived artifacts; raw loop_003 timing files are not changed.
    facts_path = LOOP3 / "measurement_facts_003.json"
    facts = read_json(facts_path)
    facts["timing_attribution"] = timing
    facts["timing_reconciliation"] = {"authoritative_value_us": authoritative, "source": str(LOOP3 / "batched_swiglu_timing.json")}
    facts_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    report_path = ROOT / "docs" / "audits" / "real_optimization_loop_003_cache_and_timing.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("BATCHED_SWIGLU_US = 55.639", f"BATCHED_SWIGLU_US = {authoritative:.3f}")
    report += "\n\n## Timing reconciliation correction\n\nThe authoritative value is `53.74621972441673 µs` from the final raw K=512 artifact. The prior `55.63945323228836 µs` value came from an earlier K=8192 diagnostic batch and is retained only as provenance, not as the final measurement.\n"
    report_path.write_text(report, encoding="utf-8")

    contract = {
        "schema": "representative_shape_contract.v1",
        "scope": "standalone authentic SwiGLU operator replay, not full nine-grid E2E",
        "fields": {
            "hidden_size": "REQUIRED",
            "ffn_hidden_size": "REQUIRED",
            "sequence_length": "REQUIRED",
            "micro_batch_size": "REQUIRED",
            "dtype": "REQUIRED",
            "tensor_parallel_size": "REQUIRED",
            "sequence_parallel": "REQUIRED",
            "gated_activation_type": "REQUIRED",
            "bias_setting": "REQUIRED",
            "layout": "REQUIRED",
            "checkpoint": "IRRELEVANT",
            "tokenizer": "IRRELEVANT",
            "dataset": "IRRELEVANT",
            "optimizer_state": "IRRELEVANT",
        },
        "replay_rule": "checkpoint/tokenizer/dataset/optimizer state do not affect isolated operator tensor shape once the required configuration fields are authoritative",
        "partition_rule_source": str(MEGATRON / "megatron/core/transformer/mlp.py"),
        "partition_rule_note": "Megatron doubles ffn_hidden_size for gated linear unit and uses tensor-parallel local construction; the target configuration values are still required.",
    }
    (OUT / "representative_shape_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    fixture = ROOT / "targets" / "megatron_5be9626" / "swiglu" / "fixtures" / "fixture_metadata.json"
    integration = ROOT / "targets" / "megatron_5be9626" / "swiglu" / "integration_contract.json"
    replay = ROOT / "targets" / "megatron_5be9626" / "swiglu" / "replay_contract.json"
    readiness = ROOT / "artifacts" / "integration" / "swiglu" / "e2e_scope" / "nine_grid_readiness.json"
    provenance = {
        "status": "PARTIAL",
        "configuration_identity": None,
        "target_scope": "requested nine-grid configuration",
        "observed_traceable_fixture_identity": "swiglu-fixture-commit-5be9626-episode2-shape-128x2x1024",
        "fixture_is_target_nine_grid": False,
        "fields": [
            {"field": "hidden_size", "value": 1024, "status": "AMBIGUOUS", **source_ref(fixture, "TRACEABLE", "Historical SwiGLU fixture; not identified as the requested nine-grid configuration.")},
            {"field": "ffn_hidden_size", "value": None, "status": "MISSING", **source_ref(replay, "TRACEABLE", "Contract provides formulas but no target configuration value.")},
            {"field": "sequence_length", "value": 128, "status": "AMBIGUOUS", **source_ref(fixture, "TRACEABLE", "Fixture metadata records sequence_length=128; target identity is unproven.")},
            {"field": "micro_batch_size", "value": 2, "status": "AMBIGUOUS", **source_ref(fixture, "TRACEABLE", "Fixture metadata records micro_batch=2; target identity is unproven.")},
            {"field": "dtype", "value": "torch.float16", "status": "AMBIGUOUS", **source_ref(fixture, "TRACEABLE", "Fixture is explicitly a historical V100 fixture.")},
            {"field": "tensor_parallel_size", "value": 1, "status": "AMBIGUOUS", **source_ref(fixture, "TRACEABLE", "Fixture metadata records tensor_parallel=1; target identity is unproven.")},
            {"field": "sequence_parallel", "value": None, "status": "MISSING", **source_ref(replay, "TRACEABLE", "Replay contract does not establish target SP setting for the fixture.")},
            {"field": "gated_activation_type", "value": "SwiGLU / SiLU gate", "status": "CONFIRMED", **source_ref(fixture, "TRACEABLE", "Fixture config records gated_linear_unit and SiLU.")},
            {"field": "bias_setting", "value": "add_bias_linear=false", "status": "CONFIRMED", **source_ref(fixture, "TRACEABLE", "Fixture config records add_bias_linear=false.")},
            {"field": "layout", "value": "contiguous row-major", "status": "CONFIRMED", **source_ref(fixture, "TRACEABLE", "Fixture records hidden-state stride/layout.")},
        ],
        "identity_failure_reason": "The available fixture is traceable but is not proven to be the requested nine-grid configuration; core FFN/SP target fields are absent under one configuration identity.",
        "nine_grid_readiness_source": source_ref(readiness, "TRACEABLE", "Existing audit explicitly says authoritative nine-grid configuration is missing."),
        "contract_sources": [source_ref(integration, "TRACEABLE", "Pinned commit and operator contract."), source_ref(replay, "TRACEABLE", "Pinned callsite and shape formulas.")],
    }
    (OUT / "nine_grid_shape_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    requirements = {
        "status": "INCOMPLETE",
        "scope": "minimum external fields needed to identify a target representative SwiGLU replay",
        "required": ["configuration_identity", "hidden_size", "ffn_hidden_size", "sequence_length", "micro_batch_size", "dtype", "tensor_parallel_size", "sequence_parallel"],
        "not_required_for_operator_replay": ["checkpoint", "tokenizer", "dataset", "optimizer_state"],
        "reason": "Those artifacts affect full training/E2E, not isolated SwiGLU tensor shape, once the required configuration fields are supplied from one authoritative identity.",
    }
    (OUT / "minimal_external_shape_requirements.json").write_text(json.dumps(requirements, indent=2), encoding="utf-8")

    matrix = {
        "scope": "representative shape/config availability only; no operator optimization",
        "operators": {
            "Dense Fused Attention": {"status": "NOT_AVAILABLE", "evidence": "D-side protocol is an external-integration template with null model_config_ref/topology/shape values.", "missing": ["configuration identity", "Sq", "Sk", "B", "heads_per_tp", "head_dim", "dtype", "strides", "backend"]},
            "Vocab-parallel Cross Entropy": {"status": "PARTIAL", "evidence": "Traceable TP=2 shape_set and dtype exist in incumbent_manifest, but no target nine-grid configuration identity.", "missing": ["target configuration identity", "global/local vocabulary contract", "rows/sequence mapping under target config"]},
            "SwiGLU": {"status": "PARTIAL", "evidence": "Traceable historical fixture supplies hidden/sequence/micro-batch/dtype/TP and layout, but not proven target identity and lacks FFN/SP fields.", "missing": requirements["required"]},
            "Residual Add RMSNorm": {"status": "PARTIAL", "evidence": "Traceable multi-shape RMSNorm suites exist, but they combine shape families and do not identify the requested nine-grid config.", "missing": ["target configuration identity", "sequence/micro-batch decomposition", "dtype/SP under target config"]},
            "MoE Grouped GEMM": {"status": "PARTIAL", "evidence": "Megatron/MoE test and campaign artifacts exist, but no same-identity target nine-grid expert/router shape contract was found.", "missing": ["target configuration identity", "experts", "tokens-per-expert distribution", "hidden/intermediate dimensions", "EP/TP"]},
        },
    }
    (OUT / "representative_operator_shape_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
