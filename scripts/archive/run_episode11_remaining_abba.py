import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent
runs = [
    ('candidate', 'B11_1b'), ('reference', 'A11_1b'),
    ('reference', 'A11_2'), ('candidate', 'B11_2'),
    ('candidate', 'B11_2b'), ('reference', 'A11_2b'),
    ('reference', 'A11_3'), ('candidate', 'B11_3'),
    ('candidate', 'B11_3b'), ('reference', 'A11_3b'),
]
results = []
for mode, run_id in runs:
    p = subprocess.run(
        [sys.executable, '_run_comparable_bench.py', '11', mode, run_id],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        text=True,
    )
    results.append({'mode': mode, 'run_id': run_id, 'returncode': p.returncode})
    print(results[-1], flush=True)
    if p.returncode != 0:
        raise SystemExit(1)
(root / 'targets/megatron_5be9626/vocab_parallel_cross_entropy/episode_11_abba_runs.json').write_text(
    __import__('json').dumps({'runs': results}, indent=2) + '\n', encoding='utf-8'
)
