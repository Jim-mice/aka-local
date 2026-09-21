import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=[]
 for p in ROOT.glob('experiments/**/record.json'):
  x=json.loads(p.read_text(encoding='utf-8')); x['_path']=str(p); rows.append(x)
 (ROOT/'indexes').mkdir(parents=True,exist_ok=True)
 (ROOT/'index.json').write_text(json.dumps({'schema_version':1,'experiments':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (ROOT/'indexes/experiments.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8')
 know=[]
 for p in ROOT.glob('knowledge/**/*.json'):
  x=json.loads(p.read_text(encoding='utf-8')); x['_path']=str(p); know.append(x)
 (ROOT/'indexes/knowledge.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in know)+('\n' if know else ''),encoding='utf-8')
 (ROOT/'index.json').write_text(json.dumps({'schema_version':1,'experiments':rows,'knowledge':know},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('experiments=',len(rows),'knowledge=',len(know))
if __name__=='__main__': main()
