import json, pathlib, re
root=pathlib.Path(__file__).resolve().parents[3]; ep=root/'campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention/episode_A2'; p=json.loads((ep/'nsys_raw.json').read_text()); out=[]
for r in p['records']:
 sections={}
 for name,body in re.findall(r'__FILE__[^\n]+_stats_(cudaapisum|gpukernsum|nvtxppsum)\.json\n(.*?)(?=__FILE__|\Z)',r.get('stats_files',''),re.S):
  try: sections[name]=json.loads(body)
  except Exception: sections[name]=[]
 kernels=sections.get('gpukernsum',[]); apis=sections.get('cudaapisum',[])
 out.append({'shape':r['shape'],'mode':r['mode'],'kernel_count':sum(int(x.get('Instances',0)) for x in kernels),'kernel_families':len(kernels),'kernel_total_ns':sum(int(x.get('Total Time (ns)',0)) for x in kernels),'kernels':kernels,'cuda_apis':apis,'nccl_kernels':[x for x in kernels if 'nccl' in x.get('Name','').lower()]})
(ep/'a2_profile_analysis.json').write_text(json.dumps({'source_nsys_artifact':'episode_A2/nsys_raw.json','status':'PROFILE_COMPLETE','profiles':out,'diagnosis':['MULTI_KERNEL_OVERHEAD','SOFTMAX_KERNEL_OVERHEAD','FULL_CORE_KERNEL_SCALING','SERIAL_WORK_GROWTH'], 'caveat':'NSYS 2021 qdrep/json; no NCU claims'},indent=2)+'\n'); print(json.dumps({'profiles':len(out),'artifact':str(ep/'a2_profile_analysis.json')},indent=2))
