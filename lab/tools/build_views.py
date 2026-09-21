import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=[json.loads(p.read_text(encoding='utf-8')) for p in ROOT.glob('experiments/**/record.json')]
 (ROOT/'views').mkdir(parents=True,exist_ok=True)
 for name,title,key in [('BY_OPERATOR','By Operator','operator'),('BY_HARDWARE','By Hardware','hardware')]:
  lines=[f'# {title}','']
  for group in sorted({r.get(key,'UNKNOWN') for r in rows}):
   lines += [f'## {group}','']+[f"- {r['id']}: {r['title']} — {r['decision']}" for r in rows if r.get(key)==group]+['']
  (ROOT/'views'/f'{name}.md').write_text('\n'.join(lines),encoding='utf-8')
 (ROOT/'views/RECENT_EXPERIMENTS.md').write_text('# Recent Experiments\n\n'+'\n'.join(f"- {r['id']}: {r['title']} [{r['decision']}]" for r in rows),encoding='utf-8')
 know=[json.loads(p.read_text(encoding='utf-8')) for p in ROOT.glob('knowledge/**/*.json')]
 for fn,title,typ in [('TECHNIQUES','Techniques',None),('ANTI_STRATEGIES','Anti-Strategies','ANTI_STRATEGY'),('OPEN_QUESTIONS','Open Questions','OPEN_QUESTION')]:
  chosen=[x for x in know if typ is None or x.get('type')==typ]
  (ROOT/'views'/f'{fn}.md').write_text(f'# {title}\n\n'+'\n'.join(f"- {x['id']}: {x.get('statement','')}" for x in chosen),encoding='utf-8')
 (ROOT/'views/LEARNING_PATH.md').write_text('# Learning Path\n\n1. Read scoped observations.\n2. Compare canonical experiments.\n3. Study backend quirks without generalizing.\n4. Run a controlled experiment only after reading the frontier.',encoding='utf-8')
 print('views built')
if __name__=='__main__': main()
