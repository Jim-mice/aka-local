import json, os, random, statistics
from pathlib import Path

base = Path('targets/megatron_5be9626/vocab_parallel_cross_entropy'); episode = int(os.environ.get('EPISODE', '7'))
refs = {b: [json.loads((base / f'comparable_reference_0_{b}.json').read_text())['rows'] for b in pair] for b, pair in {'1': ['A1','A1b'], '2': ['A2','A2b'], '3': ['A3','A3b']}.items()}
cands = {b: [json.loads((base / f'comparable_candidate_{episode}_{name}.json').read_text())['rows'] for name in names] for b, names in {'1': ['B1','B1b'], '2': ['B2','B2b'], '3': ['B3','B3b']}.items()}

def flat(runs): return [v for run in runs for row in run for v in row['distributed_samples_us']]
def rank_skew(runs): return [abs(x-y) for run in runs for row in run for x,y in zip(row['rank0']['samples_us'], row['rank1']['samples_us'])]
def stats(values):
    med = statistics.median(values); mad = statistics.median([abs(x-med) for x in values])
    return {'N': len(values), 'mean_us': statistics.mean(values), 'std_us': statistics.stdev(values), 'cv': statistics.stdev(values)/statistics.mean(values), 'median_us': med, 'min_us': min(values), 'max_us': max(values), 'MAD_us': mad}

summary = {'blocks': {}, 'all_samples': {}, 'stability': {}}
block_geo = []
for block in ['1','2','3']:
    br = [x for run in refs[block] for x in run]; bc = [x for run in cands[block] for x in run]
    ratios=[]
    for i in range(3):
        r = statistics.mean([x['mean_us'] for x in refs[block][0][i:i+1] + refs[block][1][i:i+1]])
        c = statistics.mean([x['mean_us'] for x in cands[block][0][i:i+1] + cands[block][1][i:i+1]])
        ratios.append(r/c)
    block_geo.append(statistics.geometric_mean(ratios))
    summary['blocks'][block] = {'reference': stats(flat(refs[block])), 'candidate': stats(flat(cands[block])), 'candidate_reference_speedup_per_config': ratios, 'geomean_speedup': block_geo[-1], 'reference_rank_skew_us': stats(rank_skew(refs[block])), 'candidate_rank_skew_us': stats(rank_skew(cands[block]))}
for mode, data in [('reference', refs), ('candidate', cands)]:
    allrows = [row for runs in data.values() for run in runs for row in run]
    summary['all_samples'][mode] = [stats([v for run in data.values() for runs in run for v in runs[i]['distributed_samples_us']]) for i in range(3)]
rng=random.Random(1515); boot=[]
for _ in range(10000): boot.append(statistics.mean([block_geo[rng.randrange(3)] for _ in range(3)]))
ci=[sorted(boot)[250],sorted(boot)[9750]]; qualified=all(summary['all_samples'][mode][i]['cv'] <= 0.10 for mode in ['reference','candidate'] for i in range(3)) and ci[0] > 1.0
summary['stability']={'block_geomean_speedups':block_geo,'mean_geomean_speedup':statistics.mean(block_geo),'bootstrap_ci95':ci,'qualified':qualified,'policy':'tp2_stability_v1','episode':episode}
(base/f'stability_analysis_15b7_ep{episode}.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
