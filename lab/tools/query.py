import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--operator'); p.add_argument('--hardware'); p.add_argument('--backend'); p.add_argument('--mechanism'); p.add_argument('--type'); p.add_argument('--json',action='store_true'); a=p.parse_args(); out=[]
 for f in ROOT.glob('experiments/**/record.json'):
  x=json.loads(f.read_text(encoding='utf-8')); blob=json.dumps(x).lower()
  if a.operator and x.get('operator')!=a.operator: continue
  if a.hardware and x.get('hardware')!=a.hardware: continue
  if a.backend and x.get('backend')!=a.backend: continue
  if a.mechanism and a.mechanism.lower() not in blob: continue
  if a.type and x.get('decision')!=a.type and a.type.lower() not in blob: continue
  out.append(x)
 for f in ROOT.glob('knowledge/**/*.json'):
  x=json.loads(f.read_text(encoding='utf-8')); blob=json.dumps(x).lower(); scope=json.dumps(x.get('scope',{})).lower()
  if a.operator and a.operator.lower() not in blob: continue
  if a.hardware and a.hardware.lower() not in scope: continue
  if a.backend and a.backend.lower() not in scope: continue
  if a.mechanism and a.mechanism.lower() not in blob: continue
  if a.type and x.get('type')!=a.type: continue
  out.append(x)
 print(json.dumps(out,ensure_ascii=False,indent=2) if a.json else '\n'.join(f"{x.get('id')}: {x.get('title',x.get('statement',''))} [{x.get('decision',x.get('type','UNKNOWN'))}]" for x in out))
if __name__=='__main__': main()
