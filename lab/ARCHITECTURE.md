# Architecture

```text
Execution Layer
      ↓
Campaign (operator × platform × backend × workload)
      ↓
Long-Horizon Agent
      ↓
Working Memory (live.json / journal.jsonl)
      ↓
Handoff (candidate_ready / pivot / blocked)
      ↓
Deterministic Supervisor
      ↓
Canonical Memory (memory/vNNN.json)
      ↓
Scoped Knowledge Cards
      ↓
Human Learning
```

An operator describes what is computed. A platform describes a hardware family and architecture; an execution target is a concrete machine/connection. A backend describes the implementation technology. A campaign binds all of those plus objective and workload. An episode is one bounded Agent session. An experiment is one hypothesis/change/evidence/decision unit. Working memory is provisional. Canonical memory contains observed experiment facts. Knowledge cards abstract only evidence-backed, scoped conclusions. Lessons are for the human and are not automatically prompt context.

Agent decisions are advisory. `supervisor_decision` is written only by deterministic supervisor code. Runtime failures (`AGENT_TIMEOUT`, `INFRA_FAILURE`, `INVALID_HANDOFF`) are never GPU optimization evidence.

## Provenance

This design is inspired by `/private/atrex-megatron/src/atrex-kernel-agent/long_horizon/` (read-only survey, 2026-09-15): `campaign.py`, `session.py`, `journal.py`, `protocol.py`, `store.py`, and `git_episode.py`. The lab uses the ideas but does not copy the BI-V150 gateway or remote orchestration.
