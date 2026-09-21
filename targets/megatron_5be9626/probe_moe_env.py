import pathlib, paramiko
root=pathlib.Path(__file__).resolve().parents[2]
secret=__import__('os').environ['AKA_V100_PASSWORD']
s=paramiko.SSHClient(); s.set_missing_host_key_policy(paramiko.AutoAddPolicy()); s.connect('<REMOTE_HOST>',username='<REMOTE_USER>',password=secret,timeout=20)
cmd="cd ~/aka_targets/megatron-lm-5be9626 && PYTHONPATH=$PWD <REMOTE_HOME>/venvs/lerobot-act/bin/python -c 'import importlib.util, torch; print(\"TE=\",importlib.util.find_spec(\"transformer_engine\") is not None); print(\"torch=\",torch.__version__)'"
_,o,e=s.exec_command(cmd,timeout=60); print(o.read().decode()); print(e.read().decode()); s.close()
