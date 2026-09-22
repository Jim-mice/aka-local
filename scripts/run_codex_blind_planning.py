"""Run independent read-only blind planning turns with the current Codex CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def prompt_for(case_name: str, facts: dict) -> str:
    return f"""You are performing a PLANNING_ONLY blind reasoning evaluation.
Do not edit files, run commands, benchmark, profile, implement code, add memory records, or decide promotion.
You are given only the structured facts below. Do not assume hidden historical answers.
Return one JSON object with key hypotheses, an array. Each item must contain:
hypothesis_id, generator_source=CODEX_LLM, observation, mechanism, transformation,
preconditions, expected_effect, risks, required_evidence, evidence_refs, unknowns,
estimated_magnitude (LARGE|MEDIUM|SMALL|UNKNOWN), structural_change (dataflow/lifetime/materialization/fusion/decomposition/algebra/measurement),
and asserted_facts (array of {{field, value, evidence_status}}). Do not invent measured values.
If facts are insufficient, emit a measurement hypothesis with explicit unknowns.
This is case {case_name}. The answer will be validated by a deterministic supervisor.

STRUCTURED FACTS:
{json.dumps(facts, ensure_ascii=False, sort_keys=True)}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rms = {
        "operator": "rms_norm", "dataflow": ["x_read", "row_reduction", "x_read"],
        "tensor_facts": {"x": {"immutable_between_uses": True, "consumption_count": 2, "working_set": "small_per_thread"}},
        "reduction": {"kind": "row", "ownership": "one_row_per_block", "synchronization": "tree_reduction"},
        "profile": {"global_memory_traffic": "UNKNOWN", "registers": "UNKNOWN", "kernel_launches": "UNKNOWN"},
    }
    gdn = {
        "operator": "state_update", "symbols": ["new_v", "beta", "v", "old_v", "state", "old", "k"],
        "equations": [
            "new_v = beta * v + (1 - beta) * old_v",
            "state = old + k * old_v + new_v",
            "output = state + new_v",
        ],
        "dataflow": ["old_v->new_v", "new_v->state", "new_v->output", "old->state", "k->state"],
        "profile": {"elementwise_operation_count": "UNKNOWN", "intermediate_writes": "UNKNOWN"},
    }
    swiglu_path = args.repo / "artifacts" / "integration" / "swiglu" / "authentic_profile" / "real_performance_facts_v3.json"
    swiglu = json.loads(swiglu_path.read_text(encoding="utf-8"))
    cases = {"RMSNORM_EP28_BLIND": rms, "GDN_ALGEBRA_BLIND": gdn, "SWIGLU_REAL_BLIND": swiglu}
    version = subprocess.run(["codex.exe", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    metadata = {"cli_version": version.stdout.strip(), "model_argument_supplied": False, "sandbox": "read-only", "ephemeral": True, "runs_per_case": 3}
    (args.out / "codex_protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    all_results = {}
    for case_name, facts in cases.items():
        runs = []
        for index in range(3):
            prompt = prompt_for(case_name, facts)
            completed = subprocess.run(
                ["codex.exe", "exec", "--ephemeral", "--sandbox", "read-only", "--json", "--skip-git-repo-check", "--cd", str(args.repo.resolve()), "-"],
                input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            runs.append({"run": index + 1, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
        all_results[case_name] = runs
    (args.out / "raw_llm_output.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
