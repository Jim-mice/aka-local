
"""Candidate lineage query: reads candidate_lineage.jsonl to trace the full chain."""
from __future__ import annotations

import json
from pathlib import Path


def read_lineage(episode_dir: Path) -> list[dict]:
    """Read candidate_lineage.jsonl from an episode directory."""
    path = Path(episode_dir) / "candidate_lineage.jsonl"
    if not path.is_file():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def get_lineage_chain(episode_dir: Path, candidate_id: str | None = None) -> list[dict]:
    """Return the ancestry chain for a candidate, newest-first.

    If candidate_id is None, returns the full lineage for the episode.
    """
    items = read_lineage(episode_dir)
    if not items:
        return []

    if candidate_id is None:
        return items

    # Build parent map
    parent_map: dict[str, str] = {}
    item_map: dict[str, dict] = {}
    for item in items:
        cid = item.get("candidate_id", "")
        item_map[cid] = item
        parent_map[cid] = item.get("parent_candidate_id", "")

    # Trace ancestry
    chain = []
    current = candidate_id
    seen = set()
    while current and current not in seen:
        seen.add(current)
        if current in item_map:
            chain.append(item_map[current])
            current = parent_map.get(current, "")
        else:
            break

    return chain


def format_lineage_tree(episode_dir: Path, candidate_id: str | None = None) -> str:
    """Return an ASCII tree of the candidate lineage."""
    chain = get_lineage_chain(episode_dir, candidate_id)
    if not chain:
        return "(no lineage)"

    lines = []
    for i, item in enumerate(reversed(chain)):
        indent = "  " * i
        cid = item.get("candidate_id", "?")[:20]
        exp = item.get("experiment_id", "?")
        hyp = item.get("hypothesis_id", "")[:16]
        prefix = "\u2514\u2500 " if i > 0 else ""
        lines.append(f"{indent}{prefix}{cid}  ({exp})  [{hyp}]")
    return "\n".join(lines)
