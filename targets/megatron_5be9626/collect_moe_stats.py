import pathlib,paramiko
root=pathlib.Path(__file__).resolve().parents[2]; out=root/'moe_native_sequential_expert_compute'; secret=__import__('os').environ['AKA_V100_PASSWORD']; s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
for n in ['4_4_4_4','1_3_5_7','0_0_8_8']:
 cmd=f'/usr/local/cuda-11.3/bin/nsys stats --report gpukernsum,cudaapisum,nvtxppsum --format json --output /tmp/aka_phase18a/manual_{n} /tmp/aka_phase18a/moe_{n}.qdrep'; s.exec_command(cmd,timeout=300)[1].read(); _,o,e=s.exec_command(f'for f in /tmp/aka_phase18a/manual_{n}_*; do echo __FILE__$f; cat $f; done',timeout=300); (out/f'stats_{n}.txt').write_text(o.read().decode());
s.close(); print('done')
