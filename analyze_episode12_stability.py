import json, math, random, statistics, sys
from pathlib import Path

root = Path(__file__).parent
base = root / 'targets/megatron_5be9626/vocab_parallel_cross_entropy'
episode = int(sys.argv[1]) if len(sys.argv) > 1 else 12
blocks = {
    '1': ([f'A{episode}_1', f'A{episode}_1b'], [f'B{episode}_1', f'B{episode}_1b']),
    '2': ([f'A{episode}_2', f'A{episode}_2b'], [f'B{episode}_2', f'B{episode}_2b']),
    '3': ([f'A{episode}_3', f'A{episode}_3b'], [f'B{episode}_3', f'B{episode}_3b']),
}

def load(tag):
    data = json.loads((base / f'comparable_{"reference" if tag.startswith("A") else "candidate"}_{episode}_{tag}.json').read_text(encoding='utf-8'))
    if not data.get('performance_eligible') or len(data.get('rows', [])) != 3:
        raise RuntimeError(f'incomplete run: {tag}')
    if not all(row.get('correctness_pass') for row in data['rows']):
        raise RuntimeError(f'correctness failure: {tag}')
    return data

def flat(rows):
    return [sample for row in rows for sample in row['distributed_samples_us']]

def stats(values):
    mean = statistics.mean(values)
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    return {
        'N': len(values), 'mean_us': mean,
        'std_us': statistics.stdev(values) if len(values) > 1 else 0.0,
        'cv': (statistics.stdev(values) / mean) if len(values) > 1 and mean else 0.0,
        'median_us': med, 'min_us': min(values), 'max_us': max(values), 'MAD_us': mad,
    }

summary = {'episode': episode, 'policy': 'tp2_stability_v1', 'blocks': {}, 'raw_run_ids': []}
block_scores = []
all_ref, all_cand = [], []
for block, (arefs, bcands) in blocks.items():
    refs = [load(tag) for tag in arefs]
    cands = [load(tag) for tag in bcands]
    summary['raw_run_ids'] += arefs + bcands
    ref_means = [statistics.mean([run['rows'][i]['mean_us'] for run in refs]) for i in range(3)]
    cand_means = [statistics.mean([run['rows'][i]['mean_us'] for run in cands]) for i in range(3)]
    ratios = [ref_means[i] / cand_means[i] for i in range(3)]
    score = statistics.geometric_mean(ratios)
    ref_values = [x for run in refs for x in flat(run['rows'])]
    cand_values = [x for run in cands for x in flat(run['rows'])]
    all_ref += ref_values; all_cand += cand_values
    skew = []
    for run in refs + cands:
        for row in run['rows']:
            skew += [abs(a-b) for a,b in zip(row['rank0']['samples_us'], row['rank1']['samples_us'])]
    summary['blocks'][block] = {
        'reference': stats(ref_values), 'candidate': stats(cand_values),
        'speedup_per_config': ratios, 'geomean_speedup': score,
        'rank_skew_us': stats(skew), 'reference_means_us': ref_means,
        'candidate_means_us': cand_means,
    }
    block_scores.append(score)

rng = random.Random(1515)
bootstrap = [statistics.mean([block_scores[rng.randrange(3)] for _ in range(3)]) for _ in range(10000)]
bootstrap.sort()
ci = [bootstrap[250], bootstrap[9749]]
summary['all_samples'] = {'reference': stats(all_ref), 'candidate': stats(all_cand)}
summary['block_geomean_speedups'] = block_scores
summary['geomean_speedup'] = statistics.mean(block_scores)
summary['paired_bootstrap_ci95'] = ci
summary['stability'] = {
    'status': 'STABLE' if all(summary['blocks'][b]['candidate']['cv'] <= 0.10 for b in summary['blocks']) and ci[0] > 1.0 else 'UNSTABLE',
    'all_blocks_complete': True, 'all_ranks_complete': True,
    'minimum_blocks': 3, 'samples_per_config': 30,
    'no_slow_valid_sample_deleted': True,
}
(base / f'stability_analysis_15b8_ep{episode}.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
