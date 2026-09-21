import argparse, importlib.util, json, math, statistics, sys
from pathlib import Path
import torch

ATREX=Path(r"<LOCAL_USER_HOME>\projects\atrex-bench")
def mod(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
def tm(fn, inp, warm, reps):
    for _ in range(warm): fn(**inp)
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(reps): fn(**inp)
    b.record(); b.synchronize(); return a.elapsed_time(b)/reps
def main():
    p=argparse.ArgumentParser(); p.add_argument('--incumbent',type=Path,required=True); p.add_argument('--candidate',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batches',type=int,default=5); p.add_argument('--warmup',type=int,default=20); p.add_argument('--repeats',type=int,default=100); p.add_argument('--progress-jsonl',type=Path); a=p.parse_args()
    base=mod(a.incumbent,'rms_base'); cand=mod(a.candidate,'rms_cand'); inpmod=mod(ATREX/'data/rms_norm/input.py','rms_in'); shapes=json.loads((ATREX/'data/rms_norm/shapes.json').read_text()); keys=sorted(shapes,key=lambda x:int(x)); result={k:{'input':shapes[k]['input_kwargs'],'batches':[]} for k in keys}
    if a.progress_jsonl: a.progress_jsonl.parent.mkdir(parents=True,exist_ok=True); a.progress_jsonl.write_text('',encoding='utf-8')
    # The measurements remain the established same-process A/B/B/A protocol.
    # Batch-first traversal only lets the controller report completed full
    # workload batches without inventing progress.
    for batch in range(a.batches):
        for k in keys:
            spec=shapes[k]; kw=spec['input_kwargs']; init=spec['init_kwargs']
            inp=inpmod._make_inputs(**kw); bm=base.Model(**init).cuda().eval(); cm=cand.Model(**init).cuda().eval()
            with torch.inference_mode():
                ref=bm(**inp); out=cm(**inp); diff=(out.float()-ref.float()).abs(); vals=[tm(bm,inp,a.warmup,a.repeats),tm(cm,inp,a.warmup,a.repeats),tm(cm,inp,a.warmup,a.repeats),tm(bm,inp,a.warmup,a.repeats)]
            am=(vals[0]+vals[3])/2; cv=(vals[1]+vals[2])/2; result[k]['batches'].append({'batch':batch+1,'A1_ms':vals[0],'B1_ms':vals[1],'B2_ms':vals[2],'A2_ms':vals[3],'incumbent_mean_ms':am,'candidate_mean_ms':cv,'speedup':am/cv,'max_abs':diff.max().item(),'max_rel':(diff/ref.float().abs().clamp_min(1e-12)).max().item()})
        if a.progress_jsonl:
            with a.progress_jsonl.open('a',encoding='utf-8') as f:f.write(json.dumps({'batch':batch+1,'batches':a.batches,'completed_shapes':len(keys),'shape_count':len(keys)})+'\n')
    for k in keys:
        rows=result[k]['batches']; ss=[x['speedup'] for x in rows]; result[k].update({'mean':statistics.mean(ss),'median':statistics.median(ss),'std':statistics.stdev(ss),'wins':sum(x>1 for x in ss),'n':len(ss)})
    means=[x['mean'] for x in result.values()]; ti=sum(x['incumbent_mean_ms'] for x in result.values() for x in x['batches']); tc=sum(x['candidate_mean_ms'] for x in result.values() for x in x['batches']); out={'protocol':'5 independent same-process A/B/B/A batches','warmup':a.warmup,'repeats':a.repeats,'shapes':result,'aggregate':{'arithmetic_mean_speedup':statistics.mean(means),'geometric_mean_speedup':math.prod(means)**(1/len(means)),'total_time_ratio':ti/tc},'gpu':torch.cuda.get_device_name(),'arch':list(torch.cuda.get_device_capability())}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out['aggregate'],indent=2))
if __name__=='__main__': main()
