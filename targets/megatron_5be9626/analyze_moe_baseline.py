import json,pathlib,statistics
root=pathlib.Path(__file__).resolve().parents[2]; d=root/'moe_native_sequential_expert_compute'; runs=json.loads((d/'baseline_stability_raw.json').read_text()); vals={}
for run in runs:
 obj=json.loads(run['stdout'].splitlines()[-1])
 for row in obj['benchmark']:
  k=','.join(map(str,row['distribution'])); vals.setdefault(k,[]).extend(row['samples_us'])
summary={k:{'n':len(v),'mean_us':statistics.mean(v),'std_us':statistics.stdev(v),'cv':statistics.stdev(v)/statistics.mean(v),'median_us':statistics.median(v),'min_us':min(v),'max_us':max(v)} for k,v in vals.items()}; result={'status':'REFERENCE_STABLE' if all(x['cv']<=.2 for x in summary.values()) else 'REFERENCE_STABILITY_BLOCKED','summary':summary,'policy':'moe_native_sequential_expert_compute_stability_v1'}; (d/'baseline_summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
