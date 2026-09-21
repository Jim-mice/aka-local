import base64, json, paramiko
from pathlib import Path
root=Path(__file__).resolve().parents[3]
secret=__import__('os').environ['AKA_V100_PASSWORD']
ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()); ssh.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
src=(Path(__file__).with_name('cuda118_sentinel_probe.cu')).read_bytes(); enc=base64.b64encode(src).decode()
ssh.exec_command('rm -f /tmp/aka_cuda118_sentinel_probe /tmp/aka_cuda118_sentinel_probe.cu')[1].read(); ssh.exec_command('echo '+enc+' | base64 -d > /tmp/aka_cuda118_sentinel_probe.cu')[1].read()
cmd='/usr/local/cuda-11.8/bin/nvcc -arch=sm_70 -O2 /tmp/aka_cuda118_sentinel_probe.cu -o /tmp/aka_cuda118_sentinel_probe && /tmp/aka_cuda118_sentinel_probe; echo EXIT:$?'
_,o,e=ssh.exec_command(cmd,timeout=600); result={'remote':'fallback existing project adapter','command':cmd,'stdout':o.read().decode(),'stderr':e.read().decode()}; (Path(__file__).with_name('toolchain_probe_result.json')).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); ssh.close()
