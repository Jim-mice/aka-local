"""Run the generative planner against authentic v3 evidence, planning only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(args.repo.resolve()))
    from lab.runtime.reasoning.agent_context import augment_planning_context, build_planning_context
    from lab.runtime.reasoning.hypothesis_planner import GenerativeHypothesisPlanner, NoveltyPolicy
    from lab.runtime.reasoning.mechanism_memory import MechanismStore
    from lab.runtime.reasoning.performance_model import PerformanceFacts

    raw = json.loads(args.facts.read_text(encoding="utf-8"))
    facts = PerformanceFacts(
        operator=raw["operator"], shape=dict(raw["shape"]), dtype=raw["dtype"],
        dataflow=tuple(raw.get("dataflow", [])), tensor_lifetimes=dict(raw.get("tensor_lifetimes", {})),
        global_memory_reads=tuple(raw.get("global_memory_reads", [])), global_memory_writes=tuple(raw.get("global_memory_writes", [])),
        shared_residency=tuple(raw.get("shared_residency", [])), register_residency=tuple(raw.get("register_residency", [])),
        estimated_bytes=raw.get("estimated_bytes"), estimated_flops=raw.get("estimated_flops"),
        reductions=tuple(raw.get("reductions", [])), synchronizations=tuple(raw.get("synchronizations", [])),
        kernel_launches=raw.get("kernel_launches"), producer_consumer_boundaries=tuple(raw.get("producer_consumer_boundaries", [])),
        known_reuse=tuple(raw.get("known_reuse", [])), fixed_dimensions=dict(raw.get("fixed_dimensions", {})),
        dynamic_dimensions=tuple(raw.get("dynamic_dimensions", [])), parallel_mapping=dict(raw.get("parallel_mapping", {})),
        profile_evidence=dict(raw.get("profile_evidence", {})), unknown_fields=tuple(raw.get("unknown_fields", [])),
        e2e_profile=dict(raw.get("e2e_profile", {})),
    )
    store_path = args.repo.resolve() / "knowledge" / "mechanisms.jsonl"
    mechanisms = MechanismStore(store_path).query(raw["operator"], operator=raw["operator"])
    plan = GenerativeHypothesisPlanner().plan(facts, mechanisms, novelty_policy=NoveltyPolicy())
    (args.out / "ranked_opportunities_v4.json").write_text(json.dumps(plan.to_dict(), indent=2, default=str), encoding="utf-8")
    planning_context = build_planning_context(facts, plan, mechanisms)
    (args.out / "p1_planning_context.json").write_text(json.dumps(planning_context, indent=2, default=str), encoding="utf-8")
    augmented = augment_planning_context({"snapshot": "planning-only-authentic-swiglu-v3", "context_hash": "placeholder"}, planning_context)
    (args.out / "p1_planning_context_augmented.json").write_text(json.dumps(augmented, indent=2, default=str), encoding="utf-8")
    (args.out / "planner_v4_summary.json").write_text(json.dumps({"matching_mechanisms": len(mechanisms), "raw_generation": len(plan.raw_generation), "validated": len(plan.validated), "rejected": len(plan.rejected), "ranked": len(plan.ranked), "ablation_plans": len(plan.ablation_plans), "optimization_candidate_created": False}, indent=2), encoding="utf-8")
    print(json.dumps({"matching_mechanisms": len(mechanisms), "raw_generation": len(plan.raw_generation), "validated": len(plan.validated), "rejected": len(plan.rejected), "ranked": len(plan.ranked), "ablation_plans": len(plan.ablation_plans)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
