import json,pathlib,paramiko
root=pathlib.Path(__file__).resolve().parents[2]; out=root/'moe_native_sequential_expert_compute'; secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20); runs=[]
for i in range(3):
 cmd=f'cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python /tmp/aka_phase18a/moe_expert_qualification.py'; _,o,e=s.exec_command(cmd,timeout=1200); runs.append({'invocation':i,'stdout':o.read().decode(),'stderr':e.read().decode()})
(out/'baseline_stability_raw.json').write_text(json.dumps(runs,indent=2)+'\n'); print(json.dumps({'invocations':len(runs),'artifact':str(out/'baseline_stability_raw.json')},indent=2)); s.close()
