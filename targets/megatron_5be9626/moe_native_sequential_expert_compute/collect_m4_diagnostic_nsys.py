"""Collect already-captured diagnostic-only M4 NSYS summaries."""
import json, pathlib, paramiko

root=pathlib.Path(__file__).resolve().parents[3]
ep=root/'campaigns/targets/megatron_5be9626/moe_native_sequential_expert_compute/episode_M4'
secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
d='/tmp/aka_phase19a_m4_nsys'; records=[]
for distn in ('4,4,4,4','1,3,5,7','0,0,8,8'):
 for mode in ('reference','candidate'):
  name=f'{mode}_{distn.replace(",","_")}'; cmd=f'/usr/local/cuda-11.3/bin/nsys stats --report gpukernsum,cudaapisum,nvtxppsum --format csv --output {d}/{name}_stats {d}/{name}.qdrep'
  _,o,e=s.exec_command(cmd,timeout=300); _,fo,fe=s.exec_command(f'for f in {d}/{name}_stats*; do echo __FILE__$f; cat $f; done',timeout=300)
  records.append({'distribution':distn,'mode':mode,'stats_command':cmd,'stats_stdout':o.read().decode()[-1000:],'stats_stderr':e.read().decode()[-1000:],'files':fo.read().decode()[-20000:]})
out={'status':'DIAGNOSTIC_ONLY_NOT_PROMOTION','reference':'real SequentialMLP forward boundary','candidate':'M4 custom kernel','collectives':0,'records':records}
(ep/'diagnostic_nsys_raw.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'records':len(records),'artifact':str(ep/'diagnostic_nsys_raw.json')},indent=2)); s.close()
