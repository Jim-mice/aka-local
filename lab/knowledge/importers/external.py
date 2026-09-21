"""Deterministic, provenance-first external-reference importer.

It intentionally writes normal ``lab/knowledge`` cards with
``knowledge_kind=EXTERNAL_REFERENCE``.  It never creates empirical knowledge,
does not call a model, and does not follow arbitrary outbound index links.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


def now(): return datetime.now(timezone.utc).isoformat()
def sha(value: str | bytes):
    if isinstance(value,str): value=value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()

TOPIC_RULES={
    "GPU Architecture":("architecture","compute capability","blackwell","volta"),
    "Execution Model":("thread hierarchy","thread block","grid","warp execution"),
    "Memory Hierarchy":("memory hierarchy","global memory","l2 cache"),
    "Memory Access":("coalesc","memory access","transaction"),
    "Scheduling":("scheduler","scheduling","warp scheduler"),
    "Occupancy":("occupancy","active warps"),
    "Register Pressure":("register pressure","registers per thread"),
    "Shared Memory":("shared memory","smem"),
    "Synchronization":("synchron","barrier","__syncthreads"),
    "Reduction":("reduction","reduce"),
    "Profiling":("profil","nsight","ncu"),
    "Benchmarking":("benchmark","measurement","timing"),
    "Kernel Launch":("kernel launch","launch overhead"),
    "GEMM":("gemm","matrix multiplication"),
    "Tensor Cores":("tensor core","mma"),
    "Tiling":("tiling","tile"),
    "Data Movement":("data movement","copy","pipeline"),
    "Fusion":("fusion","fused kernel"),
    "Attention":("attention","flashattention"),
    "Normalization":("normalization","rmsnorm","layernorm"),
    "RMSNorm":("rmsnorm",), "SwiGLU":("swiglu",), "MoE":("moe","mixture of experts"),
    "Triton":("triton",), "CUDA":("cuda",), "CUTLASS":("cutlass",), "CuTe":("cute","cute dsl"),
    "PyTorch Extension":("pytorch extension","torch extension"), "Numerics":("numerics","precision","rounding"),
}


def topics(text: str):
    lower=text.lower(); found=[label for label,terms in TOPIC_RULES.items() if any(term in lower for term in terms)]
    return found or ["UNCLASSIFIED"]


def scope_for(source_id: str, text: str=""):
    lower=(source_id+" "+text).lower(); scope={"vendor":[],"architecture":["generic"],"compute_capability":[],"backend":[],"language":["en"]}
    if source_id.startswith("nvidia") or source_id=="cutlass_docs": scope["vendor"]=["NVIDIA"];scope["backend"]=["CUDA"]
    if source_id=="cutlass_docs": scope["backend"]=["CUDA","CUTLASS"]
    if "triton" in lower: scope["backend"].append("Triton")
    if "blackwell" in lower: scope["architecture"]=["Blackwell"]
    elif "volta" in lower: scope["architecture"]=["Volta"]
    return scope


def authority_for(source_id: str):
    return {"modal_gpu_glossary":"COMMUNITY_CURATED","gpu_mode_resource_stream":"INDEX_ONLY",
            "gpu_mode_lectures":"COMMUNITY_MATERIAL","nvidia_cuda_programming_guide":"OFFICIAL_VENDOR",
            "nvidia_cuda_best_practices":"OFFICIAL_VENDOR","nvidia_blackwell_tuning":"OFFICIAL_VENDOR",
            "nvidia_volta_tuning":"OFFICIAL_VENDOR","cutlass_docs":"OFFICIAL_VENDOR"}.get(source_id,"COMMUNITY_MATERIAL")


def source_display_name(source_id: str) -> str:
    return {"modal_gpu_glossary":"Modal GPU Glossary", "gpu_mode_resource_stream":"GPU MODE Resource Stream",
            "gpu_mode_lectures":"GPU MODE Lectures", "nvidia_cuda_programming_guide":"NVIDIA CUDA Programming Guide",
            "nvidia_cuda_best_practices":"NVIDIA CUDA Best Practices Guide", "nvidia_blackwell_tuning":"NVIDIA Blackwell Tuning Guide",
            "nvidia_volta_tuning":"NVIDIA Volta Tuning Guide", "cutlass_docs":"NVIDIA CUTLASS Documentation"}.get(source_id,source_id.replace("_"," ").title())


def _slug(value: str): return re.sub(r"[^a-z0-9]+","-",value.lower()).strip("-")[:70] or "section"


def markdown_sections(text: str, path: str):
    """Chunk by semantic headings; only split a very large heading section."""
    lines=text.replace("\r\n","\n").split("\n"); sections=[]; stack=[]; current=[]; start=1
    def flush(end):
        nonlocal current,start
        body="\n".join(current).strip()
        if body:
            # 9000 chars is a safety boundary, not the primary chunker.
            for index in range(0,len(body),9000):
                sections.append({"heading_path":list(stack),"content":body[index:index+9000],"start":start,"end":end,"part":index//9000})
        current=[]
    for line_no,line in enumerate(lines,1):
        match=re.match(r"^(#{1,6})\s+(.+?)\s*$",line)
        if match:
            flush(line_no-1);level=len(match.group(1)); heading=match.group(2)
            stack=stack[:level-1]+[heading];start=line_no;current=[line]
        else: current.append(line)
    flush(len(lines)); return sections


class _TextHTML(HTMLParser):
    def __init__(self): super().__init__();self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in {"h1","h2","h3","h4","p","li","pre","code","section","div"}: self.parts.append("\n")
    def handle_data(self,data): self.parts.append(data)
    def text(self): return re.sub(r"\n{3,}","\n\n","".join(self.parts)).strip()


def html_sections(html: str, path: str):
    parser=_TextHTML();parser.feed(html);text=parser.text(); lines=text.splitlines(); sections=[];current=[];heading=[];start=1
    def flush(end):
        body="\n".join(current).strip()
        if body: sections.append({"heading_path":list(heading),"content":body[:9000],"start":start,"end":end,"part":0})
    for number,line in enumerate(lines,1):
        # The official pages preserve headings as standalone short lines after
        # HTML stripping.  This is intentionally conservative.
        if 2 < len(line) < 100 and not line.endswith((".",":",";")) and line==line.strip() and (line.istitle() or line.isupper()):
            flush(number-1);heading=[line];current=[line];start=number
        else: current.append(line)
    flush(len(lines));return sections or [{"heading_path":[path],"content":text[:9000],"start":1,"end":len(lines),"part":0}]


class ExternalImporter:
    def __init__(self, repo_root: Path):
        self.root=Path(repo_root);self.lab=self.root/"lab";self.sources=self.lab/"knowledge_sources";self.cards=self.lab/"knowledge"/"external_references";self.sources.mkdir(parents=True,exist_ok=True);self.cards.mkdir(parents=True,exist_ok=True)
        self.manifest_path=self.sources/"import_manifest.json";self.previous=self._read(self.manifest_path,{}) or {};self.results=[]
    @staticmethod
    def _read(path,default):
        try:return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):return default
    @staticmethod
    def _write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");temp.replace(path)
    def _git(self,*args,cwd, timeout=60):
        try:
            result=subprocess.run(["git",*args],cwd=cwd,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124,"","git command timed out"
        return result.returncode,result.stdout.strip(),result.stderr.strip()
    def snapshot_repo(self, source_id, url):
        target=self.sources/source_id
        if not target.exists():
            try:
                result=subprocess.run(["git","clone","--depth","1","--filter=blob:none",url,str(target)],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120)
            except subprocess.TimeoutExpired:
                return {"error":"git clone timed out after 120 seconds","retryable":True}
            if result.returncode:return {"error":result.stderr or result.stdout,"retryable":True}
        valid,_,error=self._git("rev-parse","--is-inside-work-tree",cwd=target)
        if valid:
            return {"error":f"incomplete source snapshot: {error or target}","retryable":True}
        # A checked-out snapshot is sufficient for deterministic import. This
        # CLI does not fetch on rerun, avoiding a blocking source refresh.
        _,commit,_=self._git("rev-parse","HEAD",cwd=target);_,branch,_=self._git("branch","--show-current",cwd=target)
        license_text="UNKNOWN"
        for name in ("LICENSE","LICENSE.md","COPYING"):
            if (target/name).exists():license_text=(target/name).read_text(encoding="utf-8",errors="replace")[:4000];break
        return {"path":target,"repo":url,"commit":commit,"branch":branch or "HEAD","license":license_text}
    def _card(self, *, source_id, source, section, source_file, source_url, kind="github_repo", index_only=False):
        content=section["content"].strip();heading=section.get("heading_path") or [Path(source_file).stem];title=heading[-1]
        content_hash=sha(content);identifier=f"external-{source_id}-{sha(source_file+'|'+str(section['start'])+'|'+content_hash)[:16]}"
        path=self.cards/source_id/f"{identifier}.json";record={
            "schema_version":1,"id":identifier,"type":"EXTERNAL_REFERENCE","knowledge_kind":"EXTERNAL_REFERENCE",
            "title":title,"statement":content[:800],"content":content,"language":"en","status":"ACTIVE","confidence":"EXTERNAL_REFERENCE",
            "evidence_for":[],"evidence_against":[],"conditions":[],"exceptions":["External reference; not empirical evidence for this Lab."],"related_knowledge":[],
            "source":{"source_id":source_id,"name":source.get("name") or source_display_name(source_id),"kind":kind,"authority":authority_for(source_id),"url":source_url,"repo":source.get("repo"),"commit":source.get("commit"),"version":source.get("version"),"license":source.get("license"),"retrieved_at":source.get("retrieved_at",now()),"index_only":index_only},
            "source_taxonomy":{"original_path":heading,"original_section":" / ".join(heading),"original_file":source_file,"original_anchor":_slug(title)},
            "canonical_taxonomy":{"domains":["GPU Performance Engineering"],"topics":topics(title+"\n"+content)},
            "scope":scope_for(source_id,title+"\n"+content),"provenance":{"content_hash":content_hash,"chunk_index":section.get("part",0),"source_start":section["start"],"source_end":section["end"]},
            "source_notes":["Imported external reference; no Lab empirical promotion."],"tags":["external-reference",authority_for(source_id).lower()],
        }
        old=self._read(path,None)
        old_source=dict((old or {}).get("source") or {}); new_source=dict(record.get("source") or {})
        old_source.pop("retrieved_at",None); new_source.pop("retrieved_at",None)
        if old and old.get("provenance",{}).get("content_hash")==content_hash and old_source==new_source:
            return "skipped",path
        self._write(path,record);return "updated" if old else "created",path
    def import_repo_markdown(self, source_id, url, *, subdir=None, include=None, index_only=False, max_documents=None):
        source=self.snapshot_repo(source_id,url)
        if source.get("error"):return {"source_id":source_id,"status":"PARTIAL","error":source["error"],"retryable":True}
        source["retrieved_at"]=now();base=source["path"]/(subdir or "");documents=created=updated=skipped=0
        if not base.is_dir():
            return {"source_id":source_id,"status":"PARTIAL","repo":url,"commit":source.get("commit"),"license":source.get("license"),"documents":0,"chunks_created":0,"chunks_updated":0,"skipped":0,"error":"CONTENT_NOT_INDEXED: source worktree is unavailable","retryable":True}
        # A subtree may carry a different license from the repository root.
        # Modal's gpu-glossary prose is CC BY 4.0 while its repository tooling
        # is MIT, so source cards must retain the narrower applicable license.
        for name in ("LICENSE", "LICENSE.md", "COPYING"):
            license_path=base/name
            if license_path.exists():
                source["license"]=license_path.read_text(encoding="utf-8",errors="replace")[:4000]
                break
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md",".markdown",".txt",".ipynb"}:continue
            relative=str(path.relative_to(source["path"])).replace("\\","/")
            if include and not include(relative):continue
            if max_documents is not None and documents >= max_documents:break
            try:
                if path.suffix.lower()==".ipynb":
                    notebook=json.loads(path.read_text(encoding="utf-8"));parts=[];heading=[]
                    for idx,cell in enumerate(notebook.get("cells",[])):
                        body="".join(cell.get("source",[])).strip()
                        if not body:continue
                        if cell.get("cell_type")=="markdown":
                            parts.extend(markdown_sections(body,relative+f"#cell-{idx}"))
                        elif cell.get("cell_type")=="code" and any(term in body.lower() for term in ("cuda","triton","cutlass","profile","kernel","rmsnorm","reduction")):
                            parts.append({"heading_path":[relative,f"code cell {idx}"],"content":"```python\n"+body+"\n```","start":idx,"end":idx,"part":0})
                else:parts=markdown_sections(path.read_text(encoding="utf-8",errors="replace"),relative)
            except (OSError,json.JSONDecodeError):continue
            documents+=1
            for section in parts:
                status,_=self._card(source_id=source_id,source=source,section=section,source_file=relative,source_url=url+"/blob/"+source["commit"]+"/"+relative,kind="community_reference" if source_id.startswith("gpu_mode") or source_id.startswith("modal") else "github_repo",index_only=index_only)
                created+=status=="created";updated+=status=="updated";skipped+=status=="skipped"
        status="PASS" if documents else "PARTIAL"
        result={"source_id":source_id,"status":status,"repo":url,"commit":source["commit"],"license":source["license"][:300],"documents":documents,"chunks_created":created,"chunks_updated":updated,"skipped":skipped,"scope":scope_for(source_id),"index_only":index_only}
        if not documents:
            result.update(error="CONTENT_NOT_INDEXED: no eligible text files in checked-out snapshot",retryable=True)
        return result
    def import_web_page(self, source_id, url, *, title=None, version=None):
        try:
            request=urllib.request.Request(url,headers={"User-Agent":"aka-local-knowledge-importer/1.0"})
            with urllib.request.urlopen(request,timeout=30) as response: html=response.read().decode("utf-8",errors="replace");final_url=response.geturl()
        except Exception as exc:return {"source_id":source_id,"status":"PARTIAL","error":f"{type(exc).__name__}: {exc}","retryable":True}
        source={"name":title or source_id,"url":final_url,"version":version,"license":"Official vendor documentation; see source terms.","retrieved_at":now()};created=updated=skipped=0;sections=html_sections(html,final_url)
        for section in sections:
            status,_=self._card(source_id=source_id,source=source,section=section,source_file=final_url,source_url=final_url,kind="official_docs")
            created+=status=="created";updated+=status=="updated";skipped+=status=="skipped"
        return {"source_id":source_id,"status":"PASS","url":final_url,"version":version,"license":source["license"],"documents":1,"chunks_created":created,"chunks_updated":updated,"skipped":skipped,"scope":scope_for(source_id)}
    def rebuild_index(self):
        records=[]
        for path in sorted((self.lab/"knowledge").glob("**/*.json")):
            item=self._read(path,None)
            if isinstance(item,dict):records.append(item)
        index=self.lab/"indexes"/"knowledge.jsonl";index.parent.mkdir(parents=True,exist_ok=True);index.write_text("".join(json.dumps(item,ensure_ascii=False)+"\n" for item in records),encoding="utf-8")
        return len(records)
    def finish(self):
        manifest={"generated_at":now(),"sources":self.results,"external_reference_count":sum(1 for path in self.cards.glob("**/*.json")),"knowledge_index_records":self.rebuild_index()};self._write(self.manifest_path,manifest)
        lines=["# External Knowledge Import Report","",f"Generated: `{manifest['generated_at']}`","", "| Source | Status | Commit / Version | Documents | Created | Updated | Skipped |", "|---|---:|---|---:|---:|---:|---:|"]
        for item in self.results:
            lines.append(f"| {item['source_id']} | {item.get('status')} | {item.get('commit') or item.get('version') or 'not detected'} | {item.get('documents',0)} | {item.get('chunks_created',0)} | {item.get('chunks_updated',0)} | {item.get('skipped',0)} |")
            lines.extend([f"  - Scope: `{json.dumps(item.get('scope',{}),ensure_ascii=True)}`",f"  - License / terms: `{str(item.get('license','UNKNOWN')).splitlines()[0][:180]}`"])
            if item.get("error"): lines.append(f"  - Import note: `{item['error']}`")
        lines += ["",f"External reference cards: `{manifest['external_reference_count']}`",f"Unified index records: `{manifest['knowledge_index_records']}`",""]
        (self.sources/"IMPORT_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
        return manifest
