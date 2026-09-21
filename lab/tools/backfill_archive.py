"""Build human-readable bilingual archives from existing structured records only."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
def main():
    n=0
    for source,record in [(p,p.parent) for p in (ROOT/"experiments").glob("**/record.json")]:
        x=json.loads(source.read_text(encoding="utf-8")); eid=x["id"]; out=ROOT/"experiment_archive"/eid; out.mkdir(parents=True,exist_ok=True)
        summary=x.get("evidence",{}).get("summary","") or x.get("observations",[""])[0]
        original=f"# {x.get('title',eid)}\n\n- ID: `{eid}`\n- Operator: `{x.get('operator')}`\n- Hardware: `{x.get('hardware')}`\n- Backend: `{x.get('backend')}`\n- Decision: `{x.get('decision')}`\n\n## Evidence\n\n{summary}\n\n## Provenance\n\n{json.dumps(x.get('provenance',{}),ensure_ascii=False,indent=2)}\n"
        zh=f"# {x.get('title',eid)}（中文归档）\n\n- 实验 ID：`{eid}`\n- 算子：`{x.get('operator')}`\n- GPU/平台：`{x.get('hardware')}`\n- Backend：`{x.get('backend')}`\n- 决策：`{x.get('decision')}`\n\n## 证据\n\n以下内容是对原始结构化证据的中文说明；数字、ID、路径和技术符号保持不变。\n\n{summary}\n\n## 来源\n\n`{x.get('provenance',{}).get('source_path')}`\n"
        analysis=f"# Structured analysis\n\nHypothesis: {x.get('hypothesis') or 'UNKNOWN'}\n\nObservations:\n"+"\n".join(f"- {v}" for v in x.get('observations',[]))+f"\n\nUnresolved questions:\n"+"\n".join(f"- {v}" for v in x.get('unresolved_questions',[]))
        atomic=lambda p,s: p.write_text(s,encoding="utf-8")
        atomic(out/"report_original.md",original); atomic(out/"report_zh-CN.md",zh); atomic(out/"analysis_original.md",analysis); atomic(out/"analysis_zh-CN.md",analysis.replace("# Structured analysis","# 结构化分析").replace("Hypothesis:","假设：").replace("Observations:","观察：").replace("Unresolved questions:","未解决问题：")); atomic(out/"metrics.json",json.dumps(x.get("performance",{}),ensure_ascii=False,indent=2)); atomic(out/"provenance.json",json.dumps({"source_path":str(source),"source_hash":sha(source),"source_record_id":eid},ensure_ascii=False,indent=2)); atomic(out/"knowledge_used.json","[]\n"); atomic(out/"knowledge_produced.json","[]\n"); atomic(out/"next_directions.json",json.dumps(x.get("next_directions",[]),ensure_ascii=False,indent=2)); atomic(out/"source_manifest.json",json.dumps({"record":str(source),"source_hash":sha(source)},ensure_ascii=False,indent=2)); n+=1
    print(f"ARCHIVE_BACKFILL_PASS records={n}")
if __name__=='__main__': main()
