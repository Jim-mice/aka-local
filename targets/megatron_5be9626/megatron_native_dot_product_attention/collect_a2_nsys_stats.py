import json, pathlib, paramiko
root=pathlib.Path(__file__).resolve().parents[3]; ep=root/'campaigns/targets/megatron_5be9626/megatron_native_dot_product_attention/episode_A2'; secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
p=json.loads((ep/'nsys_raw.json').read_text())
for r in p['records']:
 name=r['name']; cmd=f'for f in /tmp/aka_phase17b_a2/{name}_stats*; do echo __FILE__$f; cat $f; done'; _,o,e=s.exec_command(cmd,timeout=300); r['stats_files']=o.read().decode(); r['stats_files_err']=e.read().decode()
(ep/'nsys_raw.json').write_text(json.dumps(p,indent=2)+'\n'); print('collected',len(p['records'])); s.close()
