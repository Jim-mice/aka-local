import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ids=set(); errors=[]
 for p in ROOT.glob('experiments/**/record.json'):
  x=json.loads(p.read_text(encoding='utf-8')); i=x.get('id')
  if i in ids: errors.append('duplicate id '+str(i))
  ids.add(i)
 for p in ROOT.glob('knowledge/**/*.json'):
  x=json.loads(p.read_text(encoding='utf-8'))
  if x.get('type') in ('SUPPORTED_RULE','ANTI_STRATEGY') and not x.get('evidence_for'): errors.append('missing evidence '+str(p))
 if errors: print('\n'.join(errors)); return 1
 print(f'VALIDATION_PASS experiments={len(ids)}'); return 0
if __name__=='__main__': sys.exit(main())
