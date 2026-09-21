import json, math, random, statistics
from pathlib import Path
root=Path(__file__).resolve().parents[3]
ep=root/'campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention/episode_A2'
runs=json.loads((ep/'paired_raw_v2.json').read_text()); rows=[]
for invocation, run in enumerate(runs):
    for r in json.loads(run['stdout'])['rows']:
        r['global_block']=invocation*3+r['block']; rows.append(r)
configs=[(16,1),(64,2),(128,2)]; summary={}; block_ratios={}
for S,B in configs:
    key=f'{S}x{B}x1024'; summary[key]={}
    for impl in ('reference','candidate'):
        z=[r['latency_us'] for r in rows if r['S']==S and r['B']==B and r['impl']==impl]
        summary[key][impl]={'n':len(z),'mean_us':statistics.mean(z),'std_us':statistics.stdev(z),'cv':statistics.stdev(z)/statistics.mean(z),'median_us':statistics.median(z),'min_us':min(z),'max_us':max(z)}
    br=[]
    for block in range(9):
        rr=[r['latency_us'] for r in rows if r['S']==S and r['B']==B and r['impl']=='reference' and r['global_block']==block]
        cc=[r['latency_us'] for r in rows if r['S']==S and r['B']==B and r['impl']=='candidate' and r['global_block']==block]
        if len(rr)!=10 or len(cc)!=10: raise RuntimeError(f'incomplete {key} block {block}')
        br.append(statistics.mean(rr)/statistics.mean(cc))
    block_ratios[key]=br
    summary[key]['speedup']=statistics.mean([r['latency_us'] for r in rows if r['S']==S and r['B']==B and r['impl']=='reference'])/statistics.mean([r['latency_us'] for r in rows if r['S']==S and r['B']==B and r['impl']=='candidate'])
observed=math.prod(summary[k]['speedup'] for k in summary)**(1/3); rng=random.Random(1717); boot=[]
for _ in range(10000):
    vals=[]
    for k in summary:
        br=block_ratios[k]; vals.append(statistics.mean([br[rng.randrange(len(br))] for _ in br]))
    boot.append(math.prod(vals)**(1/3))
boot.sort(); result={'status':'STABILITY_QUALIFIED_BUT_PER_CONFIG_MIXED','episode':'A2','summary':summary,'block_speedups':block_ratios,'geometric_mean_speedup':observed,'bootstrap_ci95':[boot[250],boot[9750]],'bootstrap_seed':1717,'resamples':10000,'reference_stable':all(v['reference']['cv']<=.20 for v in summary.values()),'candidate_stable':all(v['candidate']['cv']<=.20 for v in summary.values()),'policy':'692f9718d1e12d922561e6d8c69cef835a37cd9c0f6e087c2ab0b342e231ae41'}
(ep/'paired_summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
