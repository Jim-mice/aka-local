import csv, json
from pathlib import Path

base = Path(__file__).parent / 'targets/megatron_5be9626/vocab_parallel_cross_entropy'

def kernel_rows(path):
    rows=[]; active=False
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('Time (%),Total Time (ns),Instances,Avg (ns)'):
            active=True; continue
        if active and line.startswith('Time (%),Total Time (ns),Num Calls'):
            break
        if active:
            try:
                row=next(csv.reader([line]));
                if len(row) >= 9: rows.append({'time_pct':float(row[0]),'total_ns':int(row[1]),'instances':int(row[2]),'name':row[8]})
            except (ValueError, StopIteration):
                pass
    return rows

def summarize(kind):
    result=[]
    for rank in (0,1):
        rows=kernel_rows(base/f'nsys_stats_{kind}_rank{rank}.txt')
        selected=lambda needle: [r for r in rows if needle in r['name']]
        local=[r for r in rows if 'local_max_kernel' in r['name'] or 'prepare_kernel' in r['name']]
        nccl=selected('ncclDevKernel_AllReduce_Sum_f32_RING_LL')
        result.append({'rank':rank,'kernel_rows':len(rows),'total_kernel_ns':sum(r['total_ns'] for r in rows),'local_candidate_kernels':{'instances':sum(r['instances'] for r in local),'total_ns':sum(r['total_ns'] for r in local)},'nccl_sum_kernels':{'instances':sum(r['instances'] for r in nccl),'total_ns':sum(r['total_ns'] for r in nccl)}})
    return result

summary={'candidate':summarize('candidate'),'reference':summarize('reference'),'source_files':['nsys_stats_candidate_rank0.txt','nsys_stats_candidate_rank1.txt','nsys_stats_reference_rank0.txt','nsys_stats_reference_rank1.txt'],'interpretation':'NSYS aggregate kernel statistics; not a replacement for the CUDA-event score and not an NCU hardware diagnosis.'}
(base/'promotion_nsys_summary_15b8.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
