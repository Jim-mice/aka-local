import json, paramiko
from pathlib import Path
root=Path(__file__).resolve().parents[2]; secret=__import__('os').environ['AKA_V100_PASSWORD']; ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()); ssh.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
rows=[]
for mode in ('forward','backward'):
    cmd=f"/usr/local/cuda-11.8/bin/nsys stats --force-export=true --report gpukernsum,cudaapisum,nvtxppsum --format json /tmp/aka_phase17a_{mode}.nsys-rep"
    _,o,e=ssh.exec_command(cmd,timeout=600); rows.append({'mode':mode,'stdout':o.read().decode(),'stderr':e.read().decode()})
(root/'targets/megatron_5be9626/attention_phase17a_nsys_stats.json').write_text(json.dumps(rows,indent=2)+'\n'); print(json.dumps(rows,indent=2)); ssh.close()
