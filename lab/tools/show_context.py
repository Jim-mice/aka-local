"""Render the compact, read-only context bundle for one campaign."""
import argparse
from lab.core.context_builder import build_context
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("campaign_id"); p.add_argument("--json",action="store_true"); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    value=build_context(root/"campaigns"/a.campaign_id)
    print(json.dumps(value, ensure_ascii=False, indent=2) if a.json else value)
if __name__ == "__main__": main()
