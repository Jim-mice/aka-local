"""Tkinter control shell for the personal GPU Operator Lab.

This GUI is an adapter over canonical records, cards and manifests.  It stores
only a pending workspace draft; selecting anything never starts an experiment.
"""
from __future__ import annotations
import json, logging, queue, re, shutil, subprocess, sys, threading, tkinter as tk, webbrowser
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, simpledialog

from .core.controller import LabController
from .core.normalization import (BACKEND_LABELS, OPERATOR_LABELS, PLATFORM_LABELS,
    display_backend, display_operator, display_platform, get_record_backend_ids,
    get_record_operator_ids, get_record_platform_ids)
from .core.persistence import atomic_json, read_json
from .ui_markdown import render_markdown

ROOT=Path(__file__).resolve().parents[1]; LAB=ROOT/"lab"
DEFAULT_CAMPAIGN="rms_norm_train__rtx5060_sm120__cuda_cpp"
RECOVERABLE_EPISODE_STATES={"PAUSED","STOPPED_RECOVERABLE","INTERRUPTED_RECOVERABLE"}

def recoverable_button_state(active,disk):
    """Derive control buttons from filesystem reconciliation, not GUI memory."""
    recoverable=bool(disk.get("resume_possible") and disk.get("state") in RECOVERABLE_EPISODE_STATES)
    return {"start_enabled":not active and not recoverable,"resume_enabled":not active and recoverable}

def _json(path, default=None):
    try: return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return default
def records():
    return [(p,x) for p in sorted((LAB/"experiments").glob("**/record.json")) if isinstance((x:=_json(p)),dict)]
def cards():
    return [(p,x) for p in sorted((LAB/"knowledge").glob("**/*.json")) if isinstance((x:=_json(p)),dict)]
def localizations(): return _json(LAB/"i18n/zh-CN/knowledge.json",{}) or {}
def _first(values, default="UNKNOWN"): return values[0] if values else default
def _ids():
    evidence=records()+cards()
    operators=set(OPERATOR_LABELS)|{p.stem for p in (LAB/"registry/operators").glob("*.yaml")}
    platforms=set(PLATFORM_LABELS)|{p.stem for p in (LAB/"registry/platforms").glob("*.yaml")}
    backends=set(BACKEND_LABELS)|{p.stem for p in (LAB/"registry/backends").glob("*.yaml")}
    for _,r in evidence:
        operators.update(get_record_operator_ids(r)); platforms.update(get_record_platform_ids(r)); backends.update(get_record_backend_ids(r))
    return sorted(filter(None,operators)),sorted(filter(None,platforms)),sorted(filter(None,backends))
def title(record,language="中文"):
    loc=localizations().get(record.get("id"),{}); original=record.get("title") or record.get("statement") or record.get("id","Unnamed")
    zh=loc.get("title_zh_CN") or loc.get("claim_zh_CN")
    return original if language=="English" else (f"{zh} / {original}" if language=="双语" and zh else zh or original)
def blob(record):
    raw=(json.dumps(record,ensure_ascii=False)+json.dumps(localizations().get(record.get("id"),{}),ensure_ascii=False)+" "+" ".join(get_record_operator_ids(record)+get_record_platform_ids(record)+get_record_backend_ids(record))).lower()
    # Search remains usable even if a legacy localization was saved with a bad
    # encoding: index technical aliases from canonical identifiers/fields.
    aliases=[]
    if "warp" in raw or "rmsnorm-v2" in str(record.get("id","")).lower():
        aliases += ["warp reduction", chr(0x5F52)+chr(0x7EA6), chr(0x626D)+chr(0x66F2)+chr(0x7EA7)+chr(0x5F52)+chr(0x7EA6)]
    if "abba" in raw:
        aliases += ["abba", "a/b/b/a", chr(0x4EA4)+chr(0x9519)+chr(0x6D4B)+chr(0x91CF)]
    if "biv150" in raw or "biv150_corex" in get_record_platform_ids(record):
        aliases += ["bi-v150", "corex", chr(0x4F0A)+chr(0x9C81)+chr(0x7EF4)+chr(0x5854)]
    return raw+" "+" ".join(aliases).lower()
def yaml_value(path,key):
    try:
        m=re.search(rf"^\s*{re.escape(key)}\s*:\s*[\"']?([^\n\"'#]+)",Path(path).read_text(encoding="utf-8"),re.M)
        return m.group(1).strip() if m else None
    except OSError:return None

def _canonical_local_path(value, *, base=None):
    """Return a case-normalized absolute local path for compatibility checks.

    Legacy manifests record the RMSNorm source as ``ops/rms_norm_v2`` while
    the Workspace Manager stores the selected Project as an absolute Windows
    path. Comparing those strings directly creates false campaign misses.
    """
    if not value:return None
    path=Path(value)
    if not path.is_absolute():path=(base or ROOT)/path
    try:path=path.resolve(strict=False)
    except OSError:path=path.absolute()
    return str(path).replace("/","\\").casefold()

def campaign_manifest(path):
    """Read either a current YAML or legacy JSON campaign manifest."""
    root=Path(path);json_path=root/"campaign.json";yaml_path=root/"campaign.yaml"
    raw=_json(json_path,{}) if json_path.exists() else None
    if isinstance(raw,dict):
        incumbent=raw.get("incumbent") or {}
        return {"id":raw.get("campaign_id") or root.name,"path":root,
                "operator_id":raw.get("operator_id") or raw.get("operator"),
                "platform_id":raw.get("platform_id") or raw.get("platform"),
                "backend_id":raw.get("backend_id") or raw.get("backend"),
                "project_id":raw.get("project_id"),
                "project_path":raw.get("project_path") or incumbent.get("source"),
                "manifest_path":json_path,"manifest_format":"json"}
    if yaml_path.exists():
        return {"id":yaml_value(yaml_path,"campaign_id") or root.name,"path":root,
                "operator_id":yaml_value(yaml_path,"operator_id") or yaml_value(yaml_path,"operator"),
                "platform_id":yaml_value(yaml_path,"platform_id") or yaml_value(yaml_path,"platform"),
                "backend_id":yaml_value(yaml_path,"backend_id") or yaml_value(yaml_path,"backend"),
                "project_id":yaml_value(yaml_path,"project_id"),"project_path":yaml_value(yaml_path,"project_path"),
                "manifest_path":yaml_path,"manifest_format":"yaml"}
    return None

def compatible_campaigns_for(draft):
    """Return campaigns matching the selected Project/Operator/Platform/Backend.

    Project identity is checked after canonicalizing relative legacy paths.
    Platform identity is compared only by canonical registry ID, never by a
    display label.
    """
    selected_path=_canonical_local_path(draft.get("project_path"));selected_project=draft.get("project_id");results=[]
    for root in sorted((LAB/"campaigns").iterdir()):
        if not root.is_dir():continue
        item=campaign_manifest(root)
        if not item or any(item.get(k)!=draft.get(k) for k in ("operator_id","platform_id","backend_id")):continue
        campaign_path=_canonical_local_path(item.get("project_path"),base=ROOT)
        project_match=(not selected_path or campaign_path==selected_path)
        if not project_match and selected_project and item.get("project_id")==selected_project and not campaign_path:project_match=True
        if project_match:results.append(item)
    return results

_EPISODE_DIRECTORY=re.compile(r"^e(\d{4})(?:_.*)?$")

def existing_episode_directories(campaign_root):
    """Return all occupied numeric episode namespaces without writing files.

    Smoke and test episodes intentionally share the same namespace as formal
    episodes, so none of them may be filtered out while allocating a proposal.
    """
    episodes=Path(campaign_root)/"episodes"
    if not episodes.is_dir():return []
    found=[]
    for entry in episodes.iterdir():
        if not entry.is_dir():continue
        match=_EPISODE_DIRECTORY.fullmatch(entry.name)
        if match:found.append((int(match.group(1)),entry.name))
    return sorted(found)

def next_episode_id(campaign_root):
    """Return the next unused eNNNN ID without creating an episode directory."""
    used={number for number,_ in existing_episode_directories(campaign_root)}
    number=max(used,default=0)+1
    while number in used:number+=1
    return f"e{number:04d}"

def advisory_agent_session(workspace):
    """Load the optional Codex adapter only when a human explicitly asks for it."""
    from .runtime.agent.codex_session import CodexAgentSession
    return CodexAgentSession(workspace)

class TextWindow(tk.Toplevel):
    def __init__(self,parent,title_,text):
        super().__init__(parent); self.title(title_); self.geometry("1050x680")
        f=ttk.Frame(self); f.pack(fill="both",expand=True); v=tk.Text(f,wrap="none",font=("Microsoft YaHei UI",10)); y=ttk.Scrollbar(f,command=v.yview); x=ttk.Scrollbar(f,orient="horizontal",command=v.xview); v.configure(yscrollcommand=y.set,xscrollcommand=x.set); v.grid(row=0,column=0,sticky="nsew"); y.grid(row=0,column=1,sticky="ns"); x.grid(row=1,column=0,sticky="ew"); f.rowconfigure(0,weight=1); f.columnconfigure(0,weight=1); render_markdown(v,text)

class KnowledgeBrowser(tk.Toplevel):
    """Global hierarchical browser: operator -> platform -> experiments/knowledge."""
    def __init__(self,parent):
        super().__init__(parent); self.title("知识库浏览器"); self.geometry("1280x800")
        self.card_data=cards(); self.experiment_data=records(); self.search_items=[]; self.node_items={}; self.search=tk.StringVar(); self.language=tk.StringVar(value="中文"); self.filters={k:tk.StringVar(value="全部") for k in ("operator","platform","backend","type","kind","source","authority","architecture","topic")}; self._build(); self.refresh_all()
    def _build(self):
        top=ttk.Frame(self,padding=8); top.pack(fill="x"); ttk.Label(top,text="搜索：").pack(side="left"); e=ttk.Entry(top,textvariable=self.search,width=32); e.pack(side="left",padx=(0,6)); e.bind("<Return>",lambda _:self.run_search()); ttk.Button(top,text="搜索",command=self.run_search).pack(side="left"); ttk.Label(top,text="语言：").pack(side="left",padx=(16,2)); lb=ttk.Combobox(top,textvariable=self.language,values=("中文","English","双语"),state="readonly",width=8); lb.pack(side="left"); lb.bind("<<ComboboxSelected>>",lambda _:self.refresh_all()); ttk.Button(top,text="打开当前来源",command=self.open_current_source).pack(side="right")
        names={"operator":"算子","platform":"GPU","backend":"Backend","type":"类型","kind":"范围","source":"来源","authority":"权威","architecture":"架构","topic":"主题"}; self.filter_boxes={}
        for key in self.filters:
            # Keep the first row compact; extended external-reference filters
            # are still real controls, simply placed on the second row.
            if key=="kind": top=ttk.Frame(self,padding=(8,0,8,6)); top.pack(fill="x")
            ttk.Label(top,text=names[key]+"：").pack(side="left",padx=(9,2)); box=ttk.Combobox(top,textvariable=self.filters[key],state="readonly",width=15); box.pack(side="left"); box.bind("<<ComboboxSelected>>",lambda _,k=key:self._filter_changed(k)); self.filter_boxes[key]=box
        self.tabs=ttk.Notebook(self); self.tabs.pack(fill="both",expand=True,padx=8,pady=(0,8)); self.map_tab=ttk.Frame(self.tabs); self.taxonomy_tab=ttk.Frame(self.tabs); self.sources_tab=ttk.Frame(self.tabs); self.results_tab=ttk.Frame(self.tabs); self.experiments_tab=ttk.Frame(self.tabs); self.archive_tab=ttk.Frame(self.tabs)
        for f,t in ((self.map_tab,"实验知识地图"),(self.taxonomy_tab,"按知识体系"),(self.sources_tab,"按来源"),(self.results_tab,"搜索结果"),(self.experiments_tab,"实验记录"),(self.archive_tab,"归档状态")):self.tabs.add(f,text=t)
        self.knowledge_tree=ttk.Treeview(self.map_tab,columns=("kind","source"),show="tree headings"); self.knowledge_tree.heading("#0",text="算子 / GPU / 内容"); self.knowledge_tree.heading("kind",text="类型"); self.knowledge_tree.heading("source",text="来源"); self.knowledge_tree.column("#0",width=480); self.knowledge_tree.column("kind",width=160); self.knowledge_tree.column("source",width=200); self.knowledge_tree.pack(side="left",fill="both",expand=True); self.knowledge_tree.bind("<<TreeviewSelect>>",self.tree_detail); self.detail_text=tk.Text(self.map_tab,wrap="word",font=("Segoe UI",10),width=55); self.detail_text.pack(side="right",fill="both",expand=True)
        self.taxonomy_tree=ttk.Treeview(self.taxonomy_tab,columns=("kind","source"),show="tree headings"); self.taxonomy_tree.heading("#0",text="统一知识体系 / 主题"); self.taxonomy_tree.heading("kind",text="身份"); self.taxonomy_tree.heading("source",text="来源"); self.taxonomy_tree.column("#0",width=600); self.taxonomy_tree.pack(side="left",fill="both",expand=True); self.taxonomy_tree.bind("<<TreeviewSelect>>",self.taxonomy_detail); self.taxonomy_detail_text=tk.Text(self.taxonomy_tab,wrap="word",font=("Segoe UI",10),width=55); self.taxonomy_detail_text.pack(side="right",fill="both",expand=True)
        self.sources_tree=ttk.Treeview(self.sources_tab,columns=("authority","scope"),show="tree headings"); self.sources_tree.heading("#0",text="外部来源 / 原始层级"); self.sources_tree.heading("authority",text="权威"); self.sources_tree.heading("scope",text="Scope"); self.sources_tree.column("#0",width=600); self.sources_tree.pack(side="left",fill="both",expand=True); self.sources_tree.bind("<<TreeviewSelect>>",self.source_detail); self.source_detail_text=tk.Text(self.sources_tab,wrap="word",font=("Segoe UI",10),width=55); self.source_detail_text.pack(side="right",fill="both",expand=True)
        self.search_tree=ttk.Treeview(self.results_tab,columns=("record","operator","platform","backend","kind","source"),show="headings")
        for k,t,w in (("record","记录",360),("operator","算子",150),("platform","GPU",160),("backend","Backend",130),("kind","类型",130),("source","来源",180)): self.search_tree.heading(k,text=t); self.search_tree.column(k,width=w)
        self.search_tree.pack(fill="both",expand=True); self.search_tree.bind("<<TreeviewSelect>>",self.search_detail); self.search_tree.bind("<Double-1>",self.open_selected)
        self.experiments_tree=ttk.Treeview(self.experiments_tab,columns=("record","operator","platform","backend","decision","source"),show="headings")
        for k,t,w in (("record","实验",330),("operator","算子",160),("platform","GPU",180),("backend","Backend",130),("decision","结果",110),("source","来源",130)):self.experiments_tree.heading(k,text=t);self.experiments_tree.column(k,width=w)
        self.experiments_tree.pack(fill="both",expand=True);self.experiments_tree.bind("<Double-1>",self.open_experiment);self.archive_text=tk.Text(self.archive_tab,font=("Segoe UI",11));self.archive_text.pack(fill="both",expand=True)
    def refresh_all(self):
        self.card_data=cards();self.experiment_data=records();self.refresh_filters();self.refresh_knowledge_tree();self.refresh_taxonomy_tree();self.refresh_sources_tree();self.refresh_experiment_table();self.run_search();self.refresh_archive_status()
    def refresh_filters(self):
        operators,platforms,backends=_ids(); external=[r for _,r in self.card_data if r.get("knowledge_kind")=="EXTERNAL_REFERENCE"]; vals={"operator":[(display_operator(x),x) for x in operators],"platform":[(display_platform(x),x) for x in platforms],"backend":[(display_backend(x),x) for x in backends],"type":sorted({str(r.get("type") or r.get("decision") or "UNKNOWN") for _,r in self.card_data+self.experiment_data}),"kind":["实验知识","外部参考"],"source":sorted({(r.get("source") or {}).get("name",r.get("source",{}).get("source_id","UNKNOWN")) for r in external}),"authority":sorted({(r.get("source") or {}).get("authority","UNKNOWN") for r in external}),"architecture":sorted({str(x) for r in external for x in (r.get("scope") or {}).get("architecture",[])}),"topic":sorted({str(x) for r in external for x in (r.get("canonical_taxonomy") or {}).get("topics",[])})}
        self.filter_lookup={k:{label:key for label,key in pairs} for k,pairs in vals.items() if k in {"operator","platform","backend"}}
        for k,box in self.filter_boxes.items():
            options=["全部"]+list(dict.fromkeys([x[0] for x in vals[k]] if k in {"operator","platform","backend"} else vals[k]));box.configure(values=options)
            if self.filters[k].get() not in options:self.filters[k].set("全部")
    def _matches(self,r):
        external=r.get("knowledge_kind")=="EXTERNAL_REFERENCE"; source=r.get("source") or {}; scope=r.get("scope") or {}; taxonomy=r.get("canonical_taxonomy") or {}; checks={"operator":[display_operator(x) for x in get_record_operator_ids(r)],"platform":[display_platform(x) for x in get_record_platform_ids(r)],"backend":[display_backend(x) for x in get_record_backend_ids(r)],"type":[str(r.get("type") or r.get("decision") or "UNKNOWN")],"kind":["外部参考" if external else "实验知识"],"source":[source.get("name",source.get("source_id","UNKNOWN"))],"authority":[source.get("authority","UNKNOWN")],"architecture":[str(x) for x in scope.get("architecture",[])],"topic":[str(x) for x in taxonomy.get("topics",[])]}
        for k,items in checks.items():
            choice=self.filters[k].get()
            if choice=="全部":continue
            if choice not in items:return False
        return True
    def _filter_changed(self,_):self.refresh_knowledge_tree();self.refresh_taxonomy_tree();self.refresh_sources_tree();self.refresh_experiment_table();self.run_search()
    def refresh_knowledge_tree(self):
        self.knowledge_tree.delete(*self.knowledge_tree.get_children());self.node_items.clear();group={}
        all_items=[("knowledge",p,r) for p,r in self.card_data]+[("experiment",p,r) for p,r in self.experiment_data]
        for kind,p,r in all_items:
            if r.get("knowledge_kind")=="EXTERNAL_REFERENCE":continue
            if not self._matches(r):continue
            for op in get_record_operator_ids(r) or ["UNKNOWN"]:
                for platform in get_record_platform_ids(r) or ["UNKNOWN"]:
                    # Presentation labels intentionally coalesce aliases such as
                    # swiglu and bias_swiglu_train without rewriting evidence IDs.
                    group.setdefault((display_operator(op),display_platform(platform),op=="UNKNOWN",platform=="UNKNOWN"),{"experiments":[],"knowledge":[]})["experiments" if kind=="experiment" else "knowledge"].append((p,r))
        by_op={}
        for (op,platform,op_unknown,platform_unknown),contents in group.items():by_op.setdefault((op,op_unknown),{})[(platform,platform_unknown)]=contents
        for (op,op_unknown) in sorted(by_op):
            on=self.knowledge_tree.insert("","end",text=op+(" ⚠ missing metadata" if op_unknown else ""),open=True)
            for (platform,platform_unknown) in sorted(by_op[(op,op_unknown)]):
                pn=self.knowledge_tree.insert(on,"end",text=platform+(" ⚠ missing metadata" if platform_unknown else ""),open=True)
                for bucket,label in (("experiments","实验"),("knowledge","知识")):
                    entries=by_op[(op,op_unknown)][(platform,platform_unknown)][bucket]
                    if not entries:continue
                    bn=self.knowledge_tree.insert(pn,"end",text=label,open=True)
                    for p,r in sorted(entries,key=lambda a:title(a[1],self.language.get())):
                         n=self.knowledge_tree.insert(bn,"end",text=title(r,self.language.get()),values=(r.get("decision") if bucket=="experiments" else r.get("type","知识"),str(p)));self.node_items[n]=(bucket,p,r)
    def _external_label(self,r):
        authority=(r.get("source") or {}).get("authority","")
        return "[官方]" if authority=="OFFICIAL_VENDOR" else ("[索引]" if authority=="INDEX_ONLY" else "[外部]")
    def refresh_taxonomy_tree(self):
        self.taxonomy_tree.delete(*self.taxonomy_tree.get_children());self.taxonomy_items={}; groups={}
        for p,r in self.card_data:
            if not self._matches(r):continue
            topics=(r.get("canonical_taxonomy") or {}).get("topics") or ["UNCLASSIFIED"]
            for topic in topics:groups.setdefault(str(topic),[]).append((p,r))
        for topic,entries in sorted(groups.items()):
            parent=self.taxonomy_tree.insert("","end",text=topic,open=True)
            for p,r in sorted(entries,key=lambda item:title(item[1],self.language.get())):
                node=self.taxonomy_tree.insert(parent,"end",text=f"{self._external_label(r) if r.get('knowledge_kind')=='EXTERNAL_REFERENCE' else '[实验]'} {title(r,self.language.get())}",values=("外部参考" if r.get("knowledge_kind")=="EXTERNAL_REFERENCE" else "实验知识",(r.get("source") or {}).get("name","本地实验")));self.taxonomy_items[node]=(p,r)
    def refresh_sources_tree(self):
        self.sources_tree.delete(*self.sources_tree.get_children());self.source_items={}; groups={}
        for p,r in self.card_data:
            if r.get("knowledge_kind")!="EXTERNAL_REFERENCE" or not self._matches(r):continue
            source=r.get("source") or {}; groups.setdefault(source.get("name",source.get("source_id","UNKNOWN")),[]).append((p,r))
        for name,entries in sorted(groups.items()):
            root=self.sources_tree.insert("","end",text=name,values=("", ""),open=True); sections={}
            for p,r in entries:sections.setdefault((r.get("source_taxonomy") or {}).get("original_section","Unclassified"),[]).append((p,r))
            for section,children in sorted(sections.items()):
                parent=self.sources_tree.insert(root,"end",text=section,open=False)
                for p,r in sorted(children,key=lambda item:title(item[1],self.language.get())):
                    s=r.get("source") or {};node=self.sources_tree.insert(parent,"end",text=title(r,self.language.get()),values=(s.get("authority","UNKNOWN"),", ".join((r.get("scope") or {}).get("architecture",[]))));self.source_items[node]=(p,r)
    def run_search(self):
        q=self.search.get().strip().lower();self.search_items=[];self.search_tree.delete(*self.search_tree.get_children())
        for kind,p,r in [("knowledge",p,r) for p,r in self.card_data]+[("experiment",p,r) for p,r in self.experiment_data]:
            if self._matches(r) and (not q or q in blob(r)):self.search_items.append((kind,p,r))
        for _,p,r in self.search_items:
            external=r.get("knowledge_kind")=="EXTERNAL_REFERENCE";source=(r.get("source") or {}).get("name","") if external else "本地实验"
            self.search_tree.insert("","end",values=(title(r,self.language.get()),display_operator(_first(get_record_operator_ids(r))),display_platform(_first(get_record_platform_ids(r))),display_backend(_first(get_record_backend_ids(r))),"外部参考" if external else r.get("type") or r.get("decision") or "UNKNOWN",source))
    def refresh_experiment_table(self):
        self.experiments_tree.delete(*self.experiments_tree.get_children());self.visible_experiments=[]
        for p,r in self.experiment_data:
            if not self._matches(r):continue
            self.visible_experiments.append((p,r));self.experiments_tree.insert("","end",values=(title(r,self.language.get()),display_operator(_first(get_record_operator_ids(r))),display_platform(_first(get_record_platform_ids(r))),display_backend(_first(get_record_backend_ids(r))),r.get("decision","UNKNOWN"),(r.get("provenance") or {}).get("source_type","UNKNOWN")))
    def refresh_archive_status(self):
        root=LAB/"experiment_archive";total=len(self.experiment_data);archived=sum((root/r.get("id","")).exists() for _,r in self.experiment_data);reports=sum((root/r.get("id","")/"report_zh-CN.md").exists() for _,r in self.experiment_data);analysis=sum((root/r.get("id","")/"analysis_zh-CN.md").exists() for _,r in self.experiment_data);localized=sum(r.get("id") in localizations() for _,r in self.card_data);self.archive_text.delete("1.0","end");self.archive_text.insert("1.0",f"实验记录总数：{total}\n已归档：{archived} / {total}\n中文报告：{reports} / {total}\n中文分析：{analysis} / {total}\n知识卡：{len(self.card_data)}\n中文知识：{localized} / {len(self.card_data)}\n")
    def tree_detail(self,_=None):
        selection=self.knowledge_tree.selection()
        if selection and selection[0] in self.node_items:_,p,r=self.node_items[selection[0]];self.show_item(p,r)
    def search_detail(self,_=None):
        selection=self.search_tree.selection()
        if selection:_,p,r=self.search_items[self.search_tree.index(selection[0])];self.show_item(p,r)
    def taxonomy_detail(self,_=None):
        selection=self.taxonomy_tree.selection()
        if selection and selection[0] in self.taxonomy_items:
            p,r=self.taxonomy_items[selection[0]];self.show_item(p,r,view=self.taxonomy_detail_text)
    def source_detail(self,_=None):
        selection=self.sources_tree.selection()
        if selection and selection[0] in self.source_items:
            p,r=self.source_items[selection[0]];self.show_item(p,r,view=self.source_detail_text)
    def show_item(self,p,r,view=None):
        view=view or self.detail_text
        if r.get("knowledge_kind")=="EXTERNAL_REFERENCE":
            source=r.get("source") or {};taxonomy=r.get("canonical_taxonomy") or {};provenance=r.get("provenance") or {};scope=r.get("scope") or {}
            self.current_source_url=source.get("url")
            body=r.get("content") or r.get("statement") or r.get("title") or ""
            text=f"{self._external_label(r)} 外部参考\n\n标题：\n{title(r,self.language.get())}\n\n正文（英文 canonical source）：\n{body}\n\n来源：\n{source.get('name')}\nAuthority：{source.get('authority')}\nURL：{source.get('url')}\nCommit / Version：{source.get('commit') or source.get('version') or '未检测'}\nLicense / Terms：{source.get('license') or '未记录'}\n\n原始层级：\n{(r.get('source_taxonomy') or {}).get('original_section')}\n原始文件：{(r.get('source_taxonomy') or {}).get('original_file')}\n\n统一主题：\n{', '.join(taxonomy.get('topics',[]))}\n\nScope：\nVendor：{', '.join(scope.get('vendor',[])) or 'generic'}\nArchitecture：{', '.join(scope.get('architecture',[])) or 'generic'}\nBackend：{', '.join(scope.get('backend',[])) or 'generic'}\n\nProvenance：\ncontent hash={provenance.get('content_hash')}\nsource lines={provenance.get('source_start')}–{provenance.get('source_end')}\n\n说明：External Reference 是背景资料，不能单独证明当前 Lab 的 GPU 瓶颈或优化结论。\n"
            render_markdown(view,text);return
        loc=localizations().get(r.get("id"),{});original=r.get("statement") or r.get("title") or "";zh=loc.get("statement_zh_CN") or loc.get("claim_zh_CN") or loc.get("title_zh_CN") or original
        text=(f"结论：\n{zh}\n\n" if self.language.get()=="中文" else (f"中文：\n{zh}\n\nEnglish / Original：\n{original}\n\n" if self.language.get()=="双语" else f"Statement:\n{original}\n\n"))
        text+=("适用范围：\n"+f"算子：{', '.join(display_operator(x) for x in get_record_operator_ids(r)) or 'UNKNOWN ⚠ missing metadata'}\n"+f"GPU：{', '.join(display_platform(x) for x in get_record_platform_ids(r)) or 'UNKNOWN ⚠ missing metadata'}\n"+f"Backend：{', '.join(display_backend(x) for x in get_record_backend_ids(r)) or 'UNKNOWN ⚠ missing metadata'}\n"+f"类型 / 决定：{r.get('type') or r.get('decision') or 'UNKNOWN'}\n"+f"证据：{', '.join(r.get('evidence_for',[])) or r.get('id','无')}\n来源：{p}\n")
        render_markdown(view,text)
    def open_current_source(self):
        url=getattr(self,"current_source_url",None)
        if not url:return messagebox.showinfo("尚未选择外部来源","请选择一条外部参考后再打开来源。",parent=self)
        webbrowser.open(url)
    def open_selected(self,_=None):
        s=self.search_tree.selection()
        if not s:return
        kind,p,r=self.search_items[self.search_tree.index(s[0])]
        if kind=="experiment":ExperimentReport(self,(p,r))
        else:messagebox.showinfo("知识卡","这是一张知识卡；其证据实验可在“实验记录”中打开。",parent=self)
    def open_experiment(self,_=None):
        s=self.experiments_tree.selection()
        if s:ExperimentReport(self,self.visible_experiments[self.experiments_tree.index(s[0])])

class ExperimentReport(tk.Toplevel):
    def __init__(self,parent,item):
        super().__init__(parent);p,r=item;self.title("实验报告 — "+title(r));self.geometry("1100x760");archive=LAB/"experiment_archive"/r.get("id","");fallback=json.dumps(r,ensure_ascii=False,indent=2);zh=(archive/"report_zh-CN.md").read_text(encoding="utf-8",errors="replace") if (archive/"report_zh-CN.md").exists() else fallback;en=(archive/"report_original.md").read_text(encoding="utf-8",errors="replace") if (archive/"report_original.md").exists() else fallback;analysis=(archive/"analysis_zh-CN.md").read_text(encoding="utf-8",errors="replace") if (archive/"analysis_zh-CN.md").exists() else "原始记录未包含公开结构化分析。";tabs=ttk.Notebook(self);tabs.pack(fill="both",expand=True,padx=8,pady=8)
        for name,text in (("概览",f"ID：{r.get('id')}\n算子：{display_operator(_first(get_record_operator_ids(r)))}\nGPU：{display_platform(_first(get_record_platform_ids(r)))}\nBackend：{display_backend(_first(get_record_backend_ids(r)))}\n决定：{r.get('decision')}"),("完整报告",zh),("分析",analysis),("Original",en),("指标",json.dumps(r.get("performance",{}),ensure_ascii=False,indent=2)),("来源",f"记录：{p}\n\n{json.dumps(r.get('provenance',{}),ensure_ascii=False,indent=2)}")):
            page=ttk.Frame(tabs);tabs.add(page,text=name);view=tk.Text(page,wrap="word",font=("Microsoft YaHei UI",10));view.pack(fill="both",expand=True);render_markdown(view,text if name not in {"指标","来源"} else "```json\n"+text+"\n```")

def inspect_project(path):
    path=Path(path); info={"path":str(path),"git_root":None,"remote":None,"head":None,"branch":None,"dirty":"UNKNOWN"}
    def git(*args):
        result=subprocess.run(["git","-C",str(path),*args],capture_output=True,text=True,encoding="utf-8",errors="replace");return result.stdout.strip() if result.returncode==0 else None
    info["git_root"]=git("rev-parse","--show-toplevel")
    if info["git_root"]:info.update(remote=git("remote","get-url","origin"),head=git("rev-parse","HEAD"),branch=git("branch","--show-current"));status=git("status","--porcelain=v1","--untracked-files=no");info["dirty"]=bool(status) if status is not None else "UNKNOWN"
    return info

def readonly_ssh_probe(host, port, user, password):
    """Run a narrowly scoped SSH probe without placing credentials in argv/files.

    The lab virtual environment intentionally has no SSH dependency.  When the
    desktop's existing Python provides Paramiko, a one-shot child receives the
    password only over stdin and returns a JSON report over stdout.
    """
    bridge = r'''import json, sys
payload=json.load(sys.stdin)
import paramiko
client=paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname=payload["host"], port=int(payload["port"]), username=payload["user"], password=payload["password"], timeout=10, look_for_keys=False, allow_agent=False)
commands={"hostname":"hostname","os":"cat /etc/os-release 2>/dev/null || uname -a","gpu":"nvidia-smi -L 2>/dev/null || true","python":"python3 --version 2>/dev/null || python --version 2>/dev/null || true","torch":"python3 -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')\" 2>/dev/null || true","compiler":"nvcc --version 2>/dev/null | tail -1 || true","disk":"df -h | head -8"}
report={}
for key, command in commands.items():
    _, out, err=client.exec_command(command, timeout=15)
    report[key]=(out.read() or err.read()).decode("utf-8", errors="replace").strip()
client.close(); print(json.dumps(report, ensure_ascii=False))
'''
    candidates=[]
    for executable in (shutil.which("python"), sys.executable):
        if executable and executable not in candidates:
            candidates.append(executable)
    failures=[]
    for executable in candidates:
        result=subprocess.run([executable,"-c",bridge],input=json.dumps({"host":host,"port":port,"user":user,"password":password}),capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=45)
        if result.returncode==0:
            return json.loads(result.stdout)
        failures.append(f"{executable}: {result.stderr.strip() or result.stdout.strip()}")
    raise RuntimeError("No available Python could perform the read-only SSH probe. " + " | ".join(failures))

class WorkspaceManager(tk.Toplevel):
    """Real local configuration: selections are a durable draft, not execution."""
    def __init__(self,parent):
        super().__init__(parent);self.parent=parent;self.controller=parent.controller;self.title("Workspace Manager");self.geometry("980x720");self.draft_path=LAB/"runtime/workspace_draft.json";self.draft=read_json(self.draft_path,{}) or {};self.draft.setdefault("operator_id","rms_norm_train");self.draft.setdefault("platform_id","rtx5060_laptop_sm120");self.draft.setdefault("backend_id","cuda_cpp");self.draft.setdefault("execution_target","local");self._build();self._render()
    def _build(self):
        n=ttk.Notebook(self);n.pack(fill="both",expand=True,padx=8,pady=8);self.hardware_page=ttk.Frame(n,padding=12);self.project_page=ttk.Frame(n,padding=12);self.operator_page=ttk.Frame(n,padding=12);self.campaign_page=ttk.Frame(n,padding=12)
        for p,t in ((self.hardware_page,"硬件 / Execution"),(self.project_page,"项目"),(self.operator_page,"算子 / Backend"),(self.campaign_page,"Campaign / Workbench")):n.add(p,text=t)
        self.hardware_summary=ttk.Label(self.hardware_page,justify="left");self.hardware_summary.pack(anchor="w",fill="x");ttk.Button(self.hardware_page,text="重新探测本地硬件",command=self.local_probe).pack(anchor="w",pady=(12,4));ttk.Button(self.hardware_page,text="使用本地 RTX 5060",command=lambda:self.use_platform("rtx5060_laptop_sm120","local")).pack(anchor="w")
        ttk.Separator(self.hardware_page).pack(fill="x",pady=14);ttk.Label(self.hardware_page,text="远端只读探测",font=("Segoe UI",11,"bold")).pack(anchor="w");ttk.Label(self.hardware_page,text="SSH 探测不创建目录、不启动实验；密码只保留于本进程。").pack(anchor="w");ttk.Button(self.hardware_page,text="连接并只读探测 SSH",command=self.remote_probe_dialog).pack(anchor="w",pady=6)
        self.project_tree=ttk.Treeview(self.project_page,columns=("kind","path","head","dirty"),show="headings");
        for k,t,w in (("kind","类型",90),("path","路径",440),("head","HEAD",160),("dirty","Dirty",80)):self.project_tree.heading(k,text=t);self.project_tree.column(k,width=w)
        self.project_tree.pack(fill="both",expand=True);b=ttk.Frame(self.project_page);b.pack(fill="x",pady=8);ttk.Button(b,text="添加本地项目",command=self.add_local_project).pack(side="left");ttk.Button(b,text="添加 Git 项目",command=self.add_git_project).pack(side="left",padx=5);ttk.Button(b,text="使用此项目",command=self.use_selected_project).pack(side="left",padx=5);ttk.Button(b,text="重新扫描",command=self._render_projects).pack(side="left",padx=5)
        ttk.Label(self.operator_page,text="当前草稿算子：").grid(row=0,column=0,sticky="w");self.operator_var=tk.StringVar();self.operator_box=ttk.Combobox(self.operator_page,textvariable=self.operator_var,state="readonly",width=45);self.operator_box.grid(row=0,column=1,sticky="w",padx=6);ttk.Button(self.operator_page,text="使用此算子",command=self.use_operator).grid(row=0,column=2,padx=6);self.operator_detail=ttk.Label(self.operator_page,justify="left");self.operator_detail.grid(row=1,column=0,columnspan=3,sticky="w",pady=10)
        ttk.Label(self.operator_page,text="Backend：").grid(row=2,column=0,sticky="w");self.backend_var=tk.StringVar();self.backend_box=ttk.Combobox(self.operator_page,textvariable=self.backend_var,state="readonly",width=45);self.backend_box.grid(row=2,column=1,sticky="w",padx=6);ttk.Button(self.operator_page,text="使用此 Backend",command=self.use_backend).grid(row=2,column=2,padx=6);a=ttk.Frame(self.operator_page);a.grid(row=3,column=0,columnspan=3,sticky="w",pady=18);ttk.Button(a,text="查看定义",command=self.show_operator_definition).pack(side="left");ttk.Button(a,text="手工添加算子",command=self.manual_operator).pack(side="left",padx=5);ttk.Button(a,text="发现新算子（只读）",command=self.discover_operator).pack(side="left",padx=5)
        self.campaign_summary=ttk.Label(self.campaign_page,justify="left");self.campaign_summary.pack(anchor="w",fill="x");self.campaign_tree=ttk.Treeview(self.campaign_page,columns=("id","operator","platform","backend"),show="headings",height=8)
        for k,t in (("id","Campaign"),("operator","算子"),("platform","GPU"),("backend","Backend")):self.campaign_tree.heading(k,text=t)
        self.campaign_tree.pack(fill="x",pady=10);b=ttk.Frame(self.campaign_page);b.pack(fill="x");ttk.Button(b,text="使用 Campaign",command=self.use_campaign).pack(side="left");ttk.Button(b,text="新建 Campaign",command=self.create_campaign).pack(side="left",padx=5);ttk.Button(b,text="提出 Workbench 方案",command=self.propose_workbench).pack(side="left",padx=5);ttk.Button(b,text="设为当前实验台",command=self.activate_workbench).pack(side="left",padx=5)
    def _save(self):atomic_json(self.draft_path,self.draft)
    def _render(self):self._render_hardware();self._render_projects();self._render_operators();self._render_campaigns()
    def _render_hardware(self):
        p=self.parent.probe or {};cc="_".join(map(str,p.get("compute_capability",[]))) or "UNKNOWN";self.hardware_summary.configure(text="已注册平台：\n"+"\n".join("• "+display_platform(x) for x in _ids()[1])+f"\n\n当前本地探测：\nGPU：{p.get('gpu_name','尚未探测')}\nArchitecture：sm_{cc}\nTorch：{p.get('torch','UNKNOWN')}\nCUDA Runtime：{p.get('torch_cuda','UNKNOWN')}\nDriver：{p.get('driver','UNKNOWN')}\nnvcc：{p.get('nvcc','UNKNOWN')}\n状态：{p.get('status','UNVERIFIED')}\nEnvironment fingerprint：{p.get('environment_fingerprint','UNKNOWN')}")
    def _projects(self):return read_json(LAB/"runtime/projects.json",[]) or []
    def _render_projects(self):
        self.project_tree.delete(*self.project_tree.get_children());self.project_items=self._projects()
        for x in self.project_items:self.project_tree.insert("","end",values=(x.get("kind"),x.get("path"),x.get("head","—"),x.get("dirty","—")))
    def _render_operators(self):
        pairs=[(display_operator(x),x) for x in _ids()[0]];self.operator_lookup={label:key for label,key in pairs};self.operator_box.configure(values=[x[0] for x in pairs]);self.operator_var.set(display_operator(self.draft["operator_id"]));pairs=[(display_backend(x),x) for x in _ids()[2]];self.backend_lookup={label:key for label,key in pairs};self.backend_box.configure(values=[x[0] for x in pairs]);self.backend_var.set(display_backend(self.draft["backend_id"]));self.operator_detail.configure(text=f"Canonical ID：{self.draft['operator_id']}\n当前项目：{self.draft.get('project_path','尚未选择')}\n选择仅更新草稿；不会创建文件夹、复制代码或启动 Agent。")
    def _render_campaigns(self):
        self.campaign_tree.delete(*self.campaign_tree.get_children());self.compatible_campaigns=compatible_campaigns_for(self.draft)
        for x in self.compatible_campaigns:self.campaign_tree.insert("","end",values=(x["id"],display_operator(x["operator_id"]),display_platform(x["platform_id"]),display_backend(x["backend_id"])))
        self.campaign_summary.configure(text=f"Project：{self.draft.get('project_path','尚未选择')}\nOperator：{display_operator(self.draft['operator_id'])}\nGPU：{display_platform(self.draft['platform_id'])}\nBackend：{display_backend(self.draft['backend_id'])}\n兼容 Campaign：{len(self.compatible_campaigns)}\n当前草稿 Campaign：{self.draft.get('campaign_id','尚未选择')}")
    def use_platform(self,platform,execution):
        if self.controller.workbench and self.controller.workbench.workbench_state=="RUNNING":return messagebox.showerror("不能切换","当前实验仍在 RUNNING；请先暂停或停止。",parent=self)
        self.draft.update(platform_id=platform,execution_target=execution);self._save();self._render()
    def local_probe(self):
        self.parent.probe=self.controller.probe_local();self.parent.refresh();self._render_hardware()
        if self.parent.probe.get("status")!="READY":messagebox.showerror("本地环境不匹配",f"GPU：{self.parent.probe.get('gpu_name')}\n状态：{self.parent.probe.get('status')}",parent=self)
    def remote_probe_dialog(self):
        d=tk.Toplevel(self);d.title("远端只读探测");d.transient(self);d.grab_set();values={n:tk.StringVar(value="22" if n=="Port" else "") for n in ("Host","Port","Username","Password")}
        for row,n in enumerate(values):ttk.Label(d,text=n+"：").grid(row=row,column=0,sticky="e",padx=8,pady=5);ttk.Entry(d,textvariable=values[n],show="*" if n=="Password" else "").grid(row=row,column=1,sticky="ew",padx=8,pady=5)
        def probe():
            host,port,user,password=(values[n].get().strip() for n in ("Host","Port","Username","Password"))
            if not host or not port or not user or not password:return messagebox.showerror("缺少字段","Host、Port、Username、Password 均必填。",parent=d)
            try:
                report=readonly_ssh_probe(host,port,user,password)
                ref=self.controller.credentials.put("ssh",{"host":host,"port":port,"user":user,"password":password});self.draft.update(execution_target="ssh",remote_host=host,remote_port=port,remote_user=user,credential_ref=ref,remote_probe=report,remote_state="WAITING_APPROVAL");self._save();d.destroy();messagebox.showinfo("REMOTE WORKBENCH CANDIDATE","只读探测完成。远端尚未批准、未创建目录、未启动实验。\n\n"+"\n\n".join(f"{k}:\n{v}" for k,v in report.items()),parent=self)
            except Exception as exc:messagebox.showerror("SSH 探测失败",f"{type(exc).__name__}: {exc}",parent=d)
        d.columnconfigure(1,weight=1);ttk.Button(d,text="连接并只读探测",command=probe).grid(row=4,column=0,columnspan=2,pady=10)
    def add_local_project(self):
        selected=filedialog.askdirectory(parent=self,title="选择本地项目目录")
        if not selected:return
        path=Path(selected);info=inspect_project(path);sid=re.sub(r"[^a-z0-9_]+","_",path.name.lower()).strip("_") or "local_project";entry={"source_id":sid,"kind":"local","registered_at":datetime.now(timezone.utc).isoformat(),**info};projects=[x for x in self._projects() if x.get("path")!=str(path)]+[entry];atomic_json(LAB/"runtime/projects.json",projects);self._render_projects();messagebox.showinfo("已注册本地项目",f"路径：{path}\nGit root：{info['git_root'] or '不是 Git repo'}\nHEAD：{info['head'] or '—'}",parent=self)
    def add_git_project(self):
        url=simpledialog.askstring("添加 Git 项目","Git Repository URL：",parent=self)
        if not url:return
        name=re.sub(r"\.git$","",url.rstrip("/")).split("/")[-1];dest=LAB/"sources"/re.sub(r"[^a-zA-Z0-9_.-]+","_",name)
        if dest.exists():return messagebox.showerror("目标已存在",f"不会覆盖：{dest}",parent=self)
        if not messagebox.askyesno("确认 clone",f"URL：{url}\n目标：{dest}\n\n将执行 git clone；不会自动 pull。是否继续？",parent=self):return
        result=subprocess.run(["git","clone",url,str(dest)],capture_output=True,text=True,encoding="utf-8",errors="replace")
        if result.returncode:return messagebox.showerror("git clone 失败",result.stderr or result.stdout,parent=self)
        entry={"source_id":dest.name,"kind":"git","registered_at":datetime.now(timezone.utc).isoformat(),**inspect_project(dest)};atomic_json(LAB/"runtime/projects.json",self._projects()+[entry]);self._render_projects()
    def use_selected_project(self):
        s=self.project_tree.selection()
        if not s:return messagebox.showwarning("未选择项目","请先在列表中选择项目。",parent=self)
        x=self.project_items[self.project_tree.index(s[0])];self.draft.update(project_id=x.get("source_id"),project_path=x.get("path"),source_commit=x.get("head"),source_dirty=x.get("dirty"));self._save();self._render_operators();self._render_campaigns()
    def use_operator(self):
        if self.controller.workbench and self.controller.workbench.workbench_state=="RUNNING":return messagebox.showerror("不能切换","当前实验仍在 RUNNING；请先暂停或停止。",parent=self)
        if (x:=self.operator_lookup.get(self.operator_var.get())):self.draft["operator_id"]=x;self.draft.pop("campaign_id",None);self._save();self._render_operators();self._render_campaigns()
    def use_backend(self):
        if (x:=self.backend_lookup.get(self.backend_var.get())):self.draft["backend_id"]=x;self.draft.pop("campaign_id",None);self._save();self._render_campaigns()
    def show_operator_definition(self):
        path=LAB/"registry/operators"/(self.draft["operator_id"]+".yaml")
        if path.exists():TextWindow(self,"算子定义 — "+display_operator(self.draft["operator_id"]),path.read_text(encoding="utf-8",errors="replace"))
        else:messagebox.showwarning("未找到定义",str(path),parent=self)
    def manual_operator(self):
        d=tk.Toplevel(self);d.title("手工添加算子 Proposal");d.transient(self);d.grab_set();fields=("显示名称","Operator ID","类型","Project","Reference path","Candidate boundary/path","Correctness command","Benchmark command","Inputs","Outputs","Dtypes","Shape parameters");entries={}
        for row,n in enumerate(fields):ttk.Label(d,text=n+"：").grid(row=row,column=0,sticky="e",padx=6,pady=3);v=tk.StringVar(value="Training" if n=="类型" else (self.draft.get("project_path","") if n=="Project" else ""));ttk.Entry(d,textvariable=v,width=65).grid(row=row,column=1,sticky="ew",padx=6,pady=3);entries[n]=v
        def approve():
            name=entries["显示名称"].get().strip();identifier=entries["Operator ID"].get().strip()
            if not name or not re.fullmatch(r"[a-z][a-z0-9_]*",identifier):return messagebox.showerror("字段无效","显示名称必填；Operator ID 必须是小写 snake_case。",parent=d)
            proposal={k:v.get().strip() for k,v in entries.items()};path=LAB/"registry/operators"/(identifier+".yaml")
            if path.exists():return messagebox.showerror("已存在",f"不会覆盖：{path}",parent=d)
            if not messagebox.askyesno("批准注册",json.dumps(proposal,ensure_ascii=False,indent=2),parent=d):return
            path.write_text(f"operator_id: {identifier}\nname: {name}\ntraining_scope: {proposal['类型']}\nsource_refs:\n  project: {proposal['Project']}\n  reference: {proposal['Reference path']}\n  candidate_boundary: {proposal['Candidate boundary/path']}\nnotes: Manual proposal; correctness={proposal['Correctness command']}; benchmark={proposal['Benchmark command']}; inputs={proposal['Inputs']}; outputs={proposal['Outputs']}; dtypes={proposal['Dtypes']}; shapes={proposal['Shape parameters']}\n",encoding="utf-8");d.destroy();self.draft["operator_id"]=identifier;self._save();self._render_operators();self._render_campaigns()
        d.columnconfigure(1,weight=1);ttk.Button(d,text="预览并批准注册",command=approve).grid(row=len(fields),column=0,columnspan=2,pady=10)
    def discover_operator(self):
        project=self.draft.get("project_path")
        if not project:return messagebox.showwarning("先选择项目","请先在“项目”页选择已注册项目。",parent=self)
        selected=filedialog.askopenfilename(parent=self,initialdir=project,title="选择要只读分析的源文件")
        if not selected:return
        source=Path(selected)
        if not source.is_file() or Path(project) not in source.parents:return messagebox.showerror("路径不允许","只能选择当前项目内部文件。",parent=self)
        if not messagebox.askyesno("只读发现",f"将把 {source} 的文本发送给 Codex 做只读 Operator Proposal。不会修改源码、编译或 benchmark。是否继续？",parent=self):return
        def work():
            prompt="Read-only operator discovery. Do not edit files, run commands, compile, benchmark, or profile. Return concise JSON keys: name, operator_id, training_scope, symbols, forward, backward, inputs, outputs, dtype, shape_parameters, reference_path, candidate_boundary, dependencies, likely_correctness_boundary, likely_benchmark_boundary.\nProject: "+project+"\nFile: "+str(source)+"\n\nSOURCE:\n"+source.read_text(encoding="utf-8",errors="replace")[:50000]
            try:s=advisory_agent_session(LAB/"runtime");s.start();reply=s.ask(prompt);s.close();answer=getattr(reply,"final_response",str(reply))
            except Exception as exc:answer=f"Discovery failed: {type(exc).__name__}: {exc}"
            self.after(0,lambda:TextWindow(self,"Operator Proposal（人工批准后才注册）",answer))
        threading.Thread(target=work,daemon=True).start()
    def use_campaign(self):
        s=self.campaign_tree.selection()
        if not s:return messagebox.showwarning("未选择 Campaign","请先选择兼容 Campaign。",parent=self)
        self.draft["campaign_id"]=self.compatible_campaigns[self.campaign_tree.index(s[0])]["id"];self._save();self._render_campaigns()
    def create_campaign(self):
        if not self.draft.get("project_path"):return messagebox.showwarning("缺少项目","新建 Campaign 前必须选择 Project。",parent=self)
        existing=compatible_campaigns_for(self.draft)
        if existing:
            names="\n".join(x["id"] for x in existing)
            return messagebox.showinfo("已有兼容 Campaign",f"当前 Project / Operator / GPU / Backend 已有兼容 Campaign：\n{names}\n\n请在列表中选择“使用 Campaign”；不会创建语义重复的 Campaign。",parent=self)
        base="{}__{}__{}".format(self.draft["operator_id"],self.draft["platform_id"],self.draft["backend_id"]);identifier=simpledialog.askstring("新建 Campaign","Campaign ID：",initialvalue=base,parent=self)
        if not identifier:return
        identifier=re.sub(r"[^a-zA-Z0-9_\-]+","_",identifier);root=LAB/"campaigns"/identifier
        if root.exists():return messagebox.showerror("已存在",f"不会覆盖：{root}",parent=self)
        preview=f"Project：{self.draft['project_path']}\nOperator：{display_operator(self.draft['operator_id'])}\nGPU：{display_platform(self.draft['platform_id'])}\nBackend：{display_backend(self.draft['backend_id'])}\nSource commit：{self.draft.get('source_commit')}\n\n将创建：{root}"
        if not messagebox.askyesno("批准创建 Campaign",preview,parent=self):return
        root.mkdir(parents=True);(root/"campaign.yaml").write_text(f"campaign_id: {identifier}\noperator_id: {self.draft['operator_id']}\nplatform_id: {self.draft['platform_id']}\nbackend_id: {self.draft['backend_id']}\nproject_path: {self.draft['project_path']}\nsource_commit: {self.draft.get('source_commit')}\nexecution_target: {self.draft.get('execution_target')}\n",encoding="utf-8");self.draft["campaign_id"]=identifier;self._save();self._render_campaigns()
    def propose_workbench(self):
        missing=[k for k in ("project_path","operator_id","platform_id","backend_id","campaign_id") if not self.draft.get(k)]
        if missing:return messagebox.showwarning("配置未完整","请先完成："+", ".join(missing),parent=self)
        campaign_root=LAB/"campaigns"/self.draft["campaign_id"];episode_id=next_episode_id(campaign_root);target=campaign_root/"episodes"/episode_id/"candidate";proposal={"proposal_id":"wb-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),"project":self.draft["project_path"],"operator":self.draft["operator_id"],"platform":self.draft["platform_id"],"backend":self.draft["backend_id"],"campaign":self.draft["campaign_id"],"episode_id":episode_id,"suggested_working_directory":str(target),"source":self.draft["project_path"],"copy_mode":"snapshot from selected project","created_at":datetime.now(timezone.utc).isoformat()};atomic_json(campaign_root/"workbench_proposal.json",proposal)
        d=tk.Toplevel(self);d.title("实验工作台方案");d.transient(self);d.grab_set();ttk.Label(d,text=f"Project：{proposal['project']}\nOperator：{display_operator(proposal['operator'])}\nGPU：{display_platform(proposal['platform'])}\nBackend：{display_backend(proposal['backend'])}\nCampaign：{proposal['campaign']}\n\nSuggested working directory：\n{proposal['suggested_working_directory']}\n\nSource：{proposal['source']}\n\n批准后才创建 candidate 快照；批准工作台不等于开始优化。",justify="left",padding=12).pack()
        def approve():
            try:
                # Reserve the episode root, rather than only ``candidate``:
                # a concurrent proposal may not share the same eNNNN ID.
                # mkdir is the final atomic non-existence check.
                target.parent.mkdir(parents=True,exist_ok=False)
                target.mkdir(exist_ok=False)
            except FileExistsError:
                d.destroy();messagebox.showerror("目录已被占用",f"不会覆盖：{target}\n\n将重新生成一个新的 Workbench Proposal。",parent=self);return self.propose_workbench()
            shutil.copytree(proposal["source"],target,dirs_exist_ok=True,ignore=shutil.ignore_patterns(".git",".venv","__pycache__","build","dist",".pytest_cache"));atomic_json(target.parent/"episode.json",{"episode_id":proposal["episode_id"],"workbench_id":proposal["proposal_id"],"working_directory":str(target),"source_directory":proposal["source"],"state":"PREPARED","created_at":datetime.now(timezone.utc).isoformat()});self.draft.update(workbench_id=proposal["proposal_id"],working_directory=str(target),workbench_state="PREPARED");self._save();self.controller.events.append("WORKBENCH_PREPARED",proposal,workbench_id=proposal["proposal_id"]);d.destroy();self.parent.active_draft=dict(self.draft);self.parent.overview();self.parent.refresh();messagebox.showinfo("工作台已创建",f"候选工作目录：\n{target}\n\nAgent 仍然 IDLE。",parent=self)
        b=ttk.Frame(d,padding=10);b.pack(fill="x");ttk.Button(b,text="批准并创建 Workbench",command=approve).pack(side="left");ttk.Button(b,text="取消",command=d.destroy).pack(side="left",padx=6)
    def activate_workbench(self):
        if not self.draft.get("working_directory") or not Path(self.draft["working_directory"]).exists():return messagebox.showwarning("尚无工作台","请先提出并批准 Workbench 方案。",parent=self)
        self.controller.configure_local(self.draft["operator_id"],self.draft["backend_id"],self.draft["campaign_id"]);self.controller.workbench.workbench_id=self.draft.get("workbench_id","local_workbench");self.controller.workbench.workbench_state="PREPARED";self.controller.events.append("WORKBENCH_PREPARED",{"draft":self.draft},workbench_id=self.controller.workbench.workbench_id);self.parent.active_draft=dict(self.draft);self.parent.overview();self.parent.refresh();messagebox.showinfo("当前实验台已更新","工作台已设为当前。Agent 仍然 IDLE。",parent=self)

def _console_category(event_type, payload):
    value=event_type or "SYSTEM"
    if value in {"KNOWLEDGE_REFRESHED","CONTEXT_BUILT"}: return "KNOWLEDGE", "知识 / Context"
    if value in {"HYPOTHESIS_RECORDED","ANALYSIS_UPDATED"}: return "ANALYSIS", "分析 / Hypothesis"
    if value in {"CANDIDATE_UPDATED","DIFF_CAPTURED"}: return "CODE", "代码 / Diff"
    if value in {"DIAGNOSTIC_STARTED","DIAGNOSTIC_ACTION_STARTED","DIAGNOSTIC_ACTION_RESULT","STATIC_EVIDENCE_COLLECTED","PROFILE_CAPABILITY","PROFILE_RESULT"}: return "DIAGNOSTIC", "诊断 / Evidence"
    if value in {"COMPILE_STARTED","COMPILE_RESULT","CORRECTNESS_STARTED","CORRECTNESS_RESULT","BENCHMARK_STARTED","BENCHMARK_RESULT"}: return "PERFORMANCE", "实验 / 性能"
    if value in {"HANDOFF_CREATED","SUPERVISOR_STARTED","SUPERVISOR_RESULT"}: return "SUPERVISOR", "Supervisor"
    if value in {"CANONICAL_MEMORY_WRITTEN","SESSION_CHECKPOINTED","RUN_STATE_SAVED"}: return "SYSTEM", "保存 / Checkpoint"
    if value in {"KNOWLEDGE_CANDIDATE_CREATED","KNOWLEDGE_LOCALIZED"}: return "KNOWLEDGE_CANDIDATE", "知识候选"
    if value=="EXPERIMENT_RECLASSIFIED": return "SYSTEM", "框架状态"
    if value in {"HUMAN_MESSAGE","HUMAN_DIRECTIVE","HUMAN_DIRECTIVE_STATUS"}: return "HUMAN", "Human"
    if value in {"RUNTIME_ERROR","RUN_INTERRUPTED","AGENT_BACKEND_ERROR"}: return "ERROR", "错误"
    if value in {"PAUSE_REQUESTED","RUN_PAUSED","RUN_RESUMED","RUN_STOPPED","STOP_REQUESTED"}: return "SYSTEM", "运行控制"
    if value=="AGENT_STATUS": return "AGENT", "Agent"
    return "SYSTEM", "系统"


_CONSOLE_LOG=logging.getLogger(__name__)

def _normalize_diagnostic_action(action):
    """Presentation-only compatibility for immutable diagnostic events.

    Early episode events stored actions as strings.  New plans use objects.
    This function never rewrites event payloads; it merely gives the Tkinter
    renderer one safe shape to consume.
    """
    if isinstance(action,str):
        return {"type":action,"legacy_format":True}
    if isinstance(action,dict):
        normalized=dict(action)
        normalized["type"]=str(normalized.get("type") or "UNKNOWN")
        return normalized
    if action is not None:
        _CONSOLE_LOG.warning("Console received unsupported diagnostic action type: %s",type(action).__name__)
    return {"type":"UNKNOWN","legacy_format":True}

def _diagnostic_action_type(action):
    return _normalize_diagnostic_action(action)["type"]

def _diagnostic_action_types(actions):
    if not isinstance(actions,list):
        _CONSOLE_LOG.warning("Console received non-list diagnostic actions payload: %s",type(actions).__name__)
        actions=[] if actions is None else [actions]
    return [_diagnostic_action_type(action) for action in actions]

def _console_payload(payload):
    if isinstance(payload,dict):
        return payload
    if payload is not None:
        _CONSOLE_LOG.warning("Console received non-object event payload: %s",type(payload).__name__)
    return {"legacy_payload":payload}


def _console_summary(event_type, payload):
    payload=_console_payload(payload)
    if event_type=="KNOWLEDGE_REFRESHED": return f"知识已刷新 {len(payload.get('knowledge_ids',[]))} 条，snapshot={str(payload.get('snapshot_hash',''))[:8]}..."
    if event_type=="HYPOTHESIS_RECORDED":
        hypothesis=_console_payload(payload.get("hypothesis") or {})
        return str(hypothesis.get('claim') or payload.get('analysis_summary') or "已记录 hypothesis")
    if event_type=="DIAGNOSTIC_STARTED": return "诊断实验开始："+", ".join(_diagnostic_action_types(payload.get("actions",[])))
    if event_type=="DIAGNOSTIC_ACTION_STARTED": return "开始诊断："+_diagnostic_action_type(payload.get("action"))
    if event_type in {"DIAGNOSTIC_ACTION_RESULT","STATIC_EVIDENCE_COLLECTED","PROFILE_CAPABILITY","PROFILE_RESULT"}:
        result=_console_payload(payload.get("result") or {})
        action=payload.get("action")
        action_name=_diagnostic_action_type(action) if action is not None else str(result.get("phase","diagnostic"))
        return f"{action_name} · {'PASS' if result.get('pass') else result.get('status','CAPABILITY_MISSING')}"
    if event_type=="AGENT_STATUS": return str(payload.get('status','Agent 状态已更新'))
    if event_type in {"HUMAN_MESSAGE","HUMAN_DIRECTIVE"}: return str(payload.get('text',''))
    if event_type=="SUPERVISOR_RESULT": return f"{payload.get('decision','UNKNOWN')}"
    if event_type=="CANONICAL_MEMORY_WRITTEN": return f"Canonical memory: {payload.get('decision','written')}"
    if event_type=="EXPERIMENT_FINISHED": return f"{payload.get('experiment_id','experiment')} · {payload.get('decision','UNKNOWN')}"
    if event_type=="EXPERIMENT_RECLASSIFIED": return f"{payload.get('experiment_id','experiment')} · FRAMEWORK FAILURE · 已从实验预算排除"
    return str(payload.get('summary') or payload.get('status') or payload.get('decision') or payload.get('reason') or event_type)


def _journal_stream_records(journal_path):
    result=[]
    for record in (_json(journal_path,[]) if str(journal_path).endswith('.json') else []): result.append(record)
    path=Path(journal_path)
    if not path.exists(): return result
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: continue
    return result


class AgentConsole(tk.Toplevel):
    """Read-only event/journal terminal plus explicit human-Agent conversation."""
    POLL_MS=800
    def __init__(self,parent):
        super().__init__(parent);self.parent=parent;self.title("Agent Console — 实时实验终端");self.geometry("1180x820");self.minsize(900,620);self.ctrl=parent.controller;self.latest_context=None;self._queue=queue.Queue();self._polling=False;self._seen_events=set();self._seen_journal=set();self._stream_items={};self._conversation=[];self.filter_var=tk.StringVar(value="全部");self.search_var=tk.StringVar();self.state_vars={key:tk.StringVar(value="—") for key in ("status","kind","bottleneck","hypothesis","action","result","next","blocker","knowledge","context")};self._build();self.refresh_context();self._load_conversation();self._start_poll();self.protocol("WM_DELETE_WINDOW",self.destroy)
    def _build(self):
        self.context_status=ttk.Label(self,justify="left",padding=8);self.context_status.pack(fill="x")
        self.tabs=ttk.Notebook(self);self.tabs.pack(fill="both",expand=True,padx=8,pady=(0,5));self.stream_page=ttk.Frame(self.tabs);self.conversation_page=ttk.Frame(self.tabs);self.state_page=ttk.Frame(self.tabs);self.context_page=ttk.Frame(self.tabs)
        for page,name in ((self.stream_page,"实验流"),(self.conversation_page,"对话"),(self.state_page,"当前状态"),(self.context_page,"Context")):self.tabs.add(page,text=name)
        self.tabs.bind("<<NotebookTabChanged>>",lambda _ : self.after_idle(self._follow_latest))
        top=ttk.Frame(self.stream_page,padding=6);top.pack(fill="x");ttk.Label(top,text="过滤：").pack(side="left");box=ttk.Combobox(top,textvariable=self.filter_var,values=("全部","Agent","实验","性能","错误","知识","Human"),state="readonly",width=12);box.pack(side="left");box.bind("<<ComboboxSelected>>",lambda _:self._refresh_stream_visibility());ttk.Label(top,text="搜索：").pack(side="left",padx=(16,2));entry=ttk.Entry(top,textvariable=self.search_var,width=28);entry.pack(side="left");entry.bind("<KeyRelease>",lambda _:self._refresh_stream_visibility());ttk.Button(top,text="回到底部",command=self._scroll_bottom).pack(side="right")
        frame=ttk.Frame(self.stream_page);frame.pack(fill="both",expand=True,padx=6,pady=(0,6));self.stream=ttk.Treeview(frame,columns=("time","category","summary"),show="headings",selectmode="browse");self.stream.heading("time",text="时间");self.stream.heading("category",text="类别");self.stream.heading("summary",text="摘要");self.stream.column("time",width=90,anchor="center");self.stream.column("category",width=150);self.stream.column("summary",width=790);self.stream.pack(side="left",fill="both",expand=True);scroll=ttk.Scrollbar(frame,command=self.stream.yview);scroll.pack(side="right",fill="y");self.stream.configure(yscrollcommand=scroll.set);self.stream.bind("<Double-1>",self._open_stream_detail)
        for tag,color in {"knowledge":"#1565c0","analysis":"#6a1b9a","code":"#007c91","pass":"#198754","warning":"#9a6a00","error":"#bb2d3b","system":"#6c757d","human":"#d76d00","candidate":"#a020a0"}.items():self.stream.tag_configure(tag,foreground=color)
        self.conversation=tk.Text(self.conversation_page,wrap="word",font=("Microsoft YaHei UI",10));self.conversation.pack(fill="both",expand=True,padx=8,pady=8)
        state=ttk.Frame(self.state_page,padding=14);state.pack(fill="both",expand=True);labels={"status":"状态","kind":"当前实验类型","bottleneck":"当前瓶颈","hypothesis":"当前 Hypothesis","action":"当前动作","result":"最新结果","next":"下一步","blocker":"Blocker","knowledge":"Knowledge snapshot","context":"Context snapshot"}
        for key,label in labels.items():ttk.Label(state,text=label+"：",font=("Microsoft YaHei UI",10,"bold")).pack(anchor="w",pady=(6,0));ttk.Label(state,textvariable=self.state_vars[key],justify="left",wraplength=980).pack(anchor="w")
        self.context_view=tk.Text(self.context_page,wrap="word",font=("Microsoft YaHei UI",10));self.context_view.pack(fill="both",expand=True,padx=8,pady=8)
        bottom=ttk.Frame(self,padding=8);bottom.pack(fill="x");self.input=ttk.Entry(bottom);self.input.pack(side="left",fill="x",expand=True);self.input.bind("<Return>",lambda _:self.ask());ttk.Button(bottom,text="ASK",command=self.ask).pack(side="left",padx=4);ttk.Button(bottom,text="DIRECTIVE",command=self.directive).pack(side="left");ttk.Button(bottom,text="查看 Context",command=lambda:self.tabs.select(self.context_page)).pack(side="left",padx=4);self.bind("<Control-f>",lambda _: (self.tabs.select(self.stream_page),self.focus_set()))
    def refresh_context(self):
        try:self.latest_context=self.ctrl.build_advisory_context(self.parent.active_draft);self._show_context_status();self._refresh_state_from_context();render_markdown(self.context_view,"```json\n"+json.dumps(self.latest_context,ensure_ascii=False,indent=2,default=str)+"\n```")
        except Exception as exc:self.context_status.configure(text=f"Context: ERROR — {type(exc).__name__}: {exc}")
    def _show_context_status(self):
        w=self.latest_context["workbench"];ep=self.latest_context.get("current_episode") or {};self.context_status.configure(text=f"Context: ATTACHED   Snapshot: {self.latest_context['context_hash'][:12]}...   Workbench: {display_platform(w.get('platform'))} · {display_operator(w.get('operator'))} · {w.get('episode') or '无 episode'}   Knowledge: {len(self.latest_context.get('relevant_knowledge',[]))}   Current episode: {ep.get('experiment_count',0)}   Canonical history: {len(self.latest_context.get('recent_canonical_experiments',[]))}")
    def _refresh_state_from_context(self):
        ep=self.latest_context.get("current_episode") or {};records=ep.get("experiments") or [];last=records[-1] if records else {};plan=(last.get("plan") or {}) if isinstance(last,dict) else {};self.state_vars["status"].set((ep.get("live_state") or {}).get("state") or "IDLE");self.state_vars["kind"].set(last.get("experiment_kind") or plan.get("experiment_kind") or "—");self.state_vars["hypothesis"].set(last.get("hypothesis") or "—");self.state_vars["result"].set(last.get("agent_result_summary") or last.get("decision") or "—");self.state_vars["next"].set(last.get("next_direction") or "等待用户操作");self.state_vars["knowledge"].set(str((self.latest_context.get("knowledge") or {}).get("count",0))+" records");self.state_vars["context"].set(self.latest_context.get("context_hash","—"))
    def _load_conversation(self):
        campaign=self.parent.active_draft.get("campaign_id",DEFAULT_CAMPAIGN);path=LAB/"runtime"/"conversations"/campaign/"advisory.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
                try:self._conversation.append(json.loads(line))
                except json.JSONDecodeError:pass
        self._render_conversation()
    def _render_conversation(self):
        parts=["# Agent Conversation\n"]
        for item in self._conversation:
            role="Human" if item.get("role")=="human" else "Agent";kind=item.get("kind","MESSAGE");parts.append(f"## {role} · {kind}\n{item.get('text','')}\n")
        render_markdown(self.conversation,"\n".join(parts));self.after_idle(self._follow_latest)
    def _start_poll(self):
        if self.winfo_exists() and not self._polling:
            self._polling=True
            def worker():
                try:
                    events=self.ctrl.events.replay();journal=Path(self.parent.active_draft.get("working_directory","")).parent/"journal.jsonl";records=_journal_stream_records(journal);live=_json(journal.parent/"live.json",{}) or {};self._queue.put((events,records,live,None))
                except Exception as exc:self._queue.put(([],[],{},exc))
                finally:self._polling=False
            threading.Thread(target=worker,daemon=True).start()
        self.after(self.POLL_MS,self._consume_poll)
    def _consume_poll(self):
        try:
            while True:
                events,records,live,error=self._queue.get_nowait()
                if error:continue
                for event in events:self._add_event(event)
                for record in records:self._add_journal(record)
                if live:self.state_vars["status"].set(live.get("state","IDLE"))
        except queue.Empty:pass
        if self.winfo_exists():self._start_poll()
    def _add_event(self,event):
        identifier=event.get("event_id")
        if not identifier or identifier in self._seen_events:return
        self._seen_events.add(identifier);typ=event.get("type","SYSTEM");payload=_console_payload(event.get("payload"));category,label=_console_category(typ,payload);item={"id":identifier,"timestamp":event.get("timestamp",""),"type":typ,"category":category,"label":label,"summary":_console_summary(typ,payload),"payload":payload,"source":"event"};self._add_stream_item(item)
        if typ=="AGENT_STATUS":self.state_vars["status"].set(str(payload.get("status","—")))
        if typ=="HYPOTHESIS_RECORDED":self.state_vars["hypothesis"].set(_console_summary(typ,payload));self.state_vars["bottleneck"].set(str(payload.get("bottleneck","—")));self.state_vars["action"].set(str(payload.get("planned_change","—")))
    def _add_journal(self,record):
        experiment=record.get("experiment_id")
        if not experiment or experiment in self._seen_journal:return
        self._seen_journal.add(experiment);timestamp=record.get("timestamp","");base={"experiment_id":experiment,"record":record,"source":"journal"};self._add_stream_item({**base,"id":experiment+":start","timestamp":timestamp,"type":"EXPERIMENT_FINISHED","category":"PERFORMANCE","label":"实验","summary":f"Experiment {experiment} finished"})
        compile_result=record.get("compile") or {};correctness=record.get("correctness") or {};bench=record.get("development_benchmark") or {};metrics=bench.get("metrics") or bench
        if compile_result:self._add_stream_item({**base,"id":experiment+":compile","timestamp":timestamp,"type":"COMPILE_RESULT","category":"PERFORMANCE","label":"编译","summary":f"{'PASS' if compile_result.get('pass') else 'FAIL'} · {compile_result.get('duration','—')}"})
        if correctness:self._add_stream_item({**base,"id":experiment+":correctness","timestamp":timestamp,"type":"CORRECTNESS_RESULT","category":"PERFORMANCE","label":"正确性","summary":f"{(correctness.get('metrics') or {}).get('tests','—')} / {(correctness.get('metrics') or {}).get('tests','—')} {'PASS' if correctness.get('pass') else 'FAIL'}"})
        if bench:self._add_stream_item({**base,"id":experiment+":bench","timestamp":timestamp,"type":"BENCHMARK_RESULT","category":"PERFORMANCE","label":"性能测试","summary":f"Development ABBA = {metrics.get('arithmetic_mean_speedup','—')}x"})
        self._add_stream_item({**base,"id":experiment+":decision","timestamp":timestamp,"type":"AGENT_DECISION","category":"AGENT","label":"决策","summary":str(record.get("decision","UNKNOWN"))})
        plan=record.get("plan") or {};interpretation=record.get("agent_interpretation") or {}
        if record.get("experiment_kind")=="DIAGNOSTIC":
            actions=record.get("diagnostic_actions") or plan.get("diagnostic_actions") or []
            self._add_stream_item({**base,"id":experiment+":diagnostic","timestamp":timestamp,"type":"DIAGNOSTIC_ACTION_RESULT","category":"DIAGNOSTIC","label":"诊断","summary":" · ".join(_diagnostic_action_types(actions)) or "DIAGNOSTIC"})
            for index,item in enumerate(record.get("diagnostic_results") or []):
                item=_console_payload(item);result=_console_payload(item.get("result") or {});self._add_stream_item({**base,"id":f"{experiment}:diagnostic:{index}","timestamp":timestamp,"type":"DIAGNOSTIC_ACTION_RESULT","category":"DIAGNOSTIC","label":"证据","summary":f"{_diagnostic_action_type(item.get('action'))} · {'PASS' if result.get('pass') else result.get('status',result.get('reason','FAIL'))}"})
        diagnostic_action_summary=" · ".join(_diagnostic_action_types(record.get("diagnostic_actions") or plan.get("diagnostic_actions") or []))
        self.state_vars["kind"].set(record.get("experiment_kind") or plan.get("experiment_kind") or "OPTIMIZATION");self.state_vars["bottleneck"].set(str(plan.get("bottleneck","—")));self.state_vars["hypothesis"].set(str((plan.get("hypothesis") or {}).get("claim") or plan.get("analysis_summary","—")));self.state_vars["action"].set(("DIAGNOSTIC: " if record.get("experiment_kind")=="DIAGNOSTIC" else "OPTIMIZATION: ")+str(diagnostic_action_summary if record.get("experiment_kind")=="DIAGNOSTIC" else plan.get("planned_change","—")));self.state_vars["result"].set(str(interpretation.get("result_summary") or record.get("decision","—")));self.state_vars["next"].set(str(interpretation.get("next_direction") or plan.get("next_direction","等待用户操作")));self.state_vars["blocker"].set(str((record.get("live") or {}).get("reason") or ("AGENT_BLOCKED" if record.get("decision")=="BLOCKED" else "—")))
    def _add_stream_item(self,item):
        self._stream_items[item["id"]]=item
        if self._matches_stream(item):self.stream.insert("","end",iid=item["id"],values=(item["timestamp"].replace("T"," ")[11:19] if len(item["timestamp"])>10 else item["timestamp"],item["label"],item["summary"]),tags=(self._tag_for(item),))
        self.after_idle(self._follow_latest)
    def _tag_for(self,item):
        typ=item["type"];summary=item["summary"].upper()
        if "FAIL" in summary or item["category"]=="ERROR":return "error"
        if any(x in summary for x in ("PASS","PROMOTE","ACCEPT")):return "pass"
        if any(x in summary for x in ("BLOCKED","REJECT","WARNING","ROBUSTNESS")):return "warning"
        return {"KNOWLEDGE":"knowledge","ANALYSIS":"analysis","CODE":"code","HUMAN":"human","KNOWLEDGE_CANDIDATE":"candidate","SYSTEM":"system","AGENT":"analysis","PERFORMANCE":"code","DIAGNOSTIC":"knowledge","SUPERVISOR":"system"}.get(item["category"],"system")
    def _matches_stream(self,item):
        choice=self.filter_var.get();mapping={"Agent":{"AGENT","ANALYSIS"},"实验":{"PERFORMANCE","DIAGNOSTIC","SUPERVISOR"},"性能":{"PERFORMANCE"},"错误":{"ERROR"},"知识":{"KNOWLEDGE","KNOWLEDGE_CANDIDATE"},"Human":{"HUMAN"}}
        if choice!="全部" and item["category"] not in mapping.get(choice,set()):return False
        return self.search_var.get().strip().lower() in (item["summary"]+" "+item["type"]+" "+item["label"]).lower()
    def _refresh_stream_visibility(self):
        self.stream.delete(*self.stream.get_children())
        for item in self._stream_items.values():
            if self._matches_stream(item):self.stream.insert("","end",iid=item["id"],values=(item["timestamp"].replace("T"," ")[11:19] if len(item["timestamp"])>10 else item["timestamp"],item["label"],item["summary"]),tags=(self._tag_for(item),))
        self.after_idle(self._follow_latest)
    def _follow_latest(self):
        if not self.winfo_exists():return
        children=self.stream.get_children()
        if children:self.stream.see(children[-1]);self.stream.yview_moveto(1.0)
        # Tk's final implicit newline is not always scrollable as ``end``.
        # Target the final actual character so long Markdown answers finish
        # visibly at their last rendered line.
        try:self.conversation.see("end-1c");self.conversation.yview_moveto(1.0)
        except tk.TclError:pass
    def _pause_autoscroll(self,_=None):pass
    def _scroll_bottom(self):self._follow_latest()
    def _open_stream_detail(self,_=None):
        selected=self.stream.selection()
        if not selected:return
        item=self._stream_items[selected[0]];detail=tk.Toplevel(self);detail.title(item["label"]+" — "+item["summary"]);detail.geometry("900x620");view=tk.Text(detail,wrap="word",font=("Microsoft YaHei UI",10));view.pack(fill="both",expand=True,padx=8,pady=8);payload=item.get("record") or item.get("payload") or {};render_markdown(view,f"# {item['label']}\n\n**时间：** {item['timestamp']}\n\n**摘要：** {item['summary']}\n\n```json\n{json.dumps(payload,ensure_ascii=False,indent=2,default=str)}\n```");buttons=ttk.Frame(detail,padding=8);buttons.pack(fill="x")
        if item.get("source")=="journal":ttk.Button(buttons,text="打开实验报告",command=lambda:TextWindow(detail,"Experiment — "+item["experiment_id"],"```json\n"+json.dumps(item["record"],ensure_ascii=False,indent=2)+"\n```")).pack(side="left")
        if item["category"] in {"KNOWLEDGE","KNOWLEDGE_CANDIDATE"}:ttk.Button(buttons,text="打开知识库",command=lambda:KnowledgeBrowser(self.parent)).pack(side="left",padx=5)
    def ask(self):
        question=self.input.get().strip()
        if not question:return
        self._conversation.append({"role":"human","kind":"ASK","text":question});self._render_conversation();self.tabs.select(self.conversation_page);self.input.delete(0,"end")
        def work():
            try:result=self.ctrl.ask_advisory(question,self.parent.active_draft);answer=result["answer"];context=result["context"]
            except Exception as exc:answer=f"# Agent backend error\n\n`{type(exc).__name__}: {exc}`";context=None
            self.after(0,lambda:(self._conversation.append({"role":"agent","kind":"ANSWER","text":answer}),setattr(self,"latest_context",context or self.latest_context),self._render_conversation(),self._show_context_status() if self.latest_context else None))
        threading.Thread(target=work,daemon=True).start()
    def directive(self):
        text=self.input.get().strip()
        if text:self.ctrl.directive(text);self._conversation.append({"role":"human","kind":"DIRECTIVE","text":f"**PENDING**\n\n{text}"});self._render_conversation();self.tabs.select(self.conversation_page);self.input.delete(0,"end")

class App(tk.Tk):
    def __init__(self):
        super().__init__();self.title("GPU Operator Lab");self.geometry("1200x800");self.controller=LabController(ROOT);self.probe=None;self.episode_state=None;self.active_draft=read_json(LAB/"runtime/workspace_draft.json",{}) or {};self._build();self.after(100,self.probe_local);self.after(500,self.recovery_scan);self.after(1000,self.refresh)
    def _build(self):
        top=ttk.Frame(self,padding=10);top.pack(fill="x");ttk.Label(top,text="GPU OPERATOR LAB",font=("Segoe UI",17,"bold")).pack(side="left")
        for label,fn in (("KNOWLEDGE",lambda:KnowledgeBrowser(self)),("AGENT CONSOLE",lambda:AgentConsole(self)),("CHANGE WORKBENCH",lambda:WorkspaceManager(self)),("NEW CAMPAIGN",lambda:WorkspaceManager(self)),("HELP",self.help)):ttk.Button(top,text=label,command=fn).pack(side="right",padx=3)
        self.banner=ttk.Label(self,padding=10,justify="left");self.banner.pack(fill="x");self.main=ttk.Frame(self,padding=14);self.main.pack(fill="both",expand=True);self.overview()
    def probe_local(self):
        def work():self.probe=self.controller.probe_local();self.after(0,self.refresh)
        threading.Thread(target=work,daemon=True).start()
    def recovery_scan(self):
        runs=self.controller.recoverable_runs()
        if not runs:return
        run=runs[0];text=f"发现未完成实验\n\nRun ID：{run.get('run_id')}\nCampaign：{run.get('campaign_id')}\nEpisode：{run.get('episode_id')}\n状态：{run.get('state')}\n最后安全点：{run.get('last_safe_point')}\nCandidate：{run.get('candidate_root')}\n\n恢复会重新 probe 环境并创建新的 Codex session，不依赖旧对话。是否恢复？"
        if messagebox.askyesno("发现未完成实验",text,parent=self):
            try:self.controller.resume_recovered(run.get('run_id'));self.overview();self.refresh()
            except Exception as exc:messagebox.showerror("恢复被阻止",str(exc),parent=self)
    def clear(self):
        for w in self.main.winfo_children():w.destroy()
    def card(self,name,rows):
        f=ttk.LabelFrame(self.main,text=name,padding=12);f.pack(fill="x",pady=6)
        for k,v in rows:ttk.Label(f,text=f"{k}: {v}").pack(anchor="w",pady=1)
    def overview(self):
        self.clear();d=self.active_draft;status=self.controller.status();active=status.get("runner_active",False)
        try:disk=self.controller.reconcile_episode_state(d)
        except Exception:disk={"state":d.get("workbench_state","NOT CREATED"),"experiment_count":0,"max_experiments":5,"resume_possible":False}
        state="RUNNING" if active else disk.get("state","NOT CREATED")
        self.episode_state=None if active else state
        buttons=recoverable_button_state(active,disk)
        self.card("当前实验台",[("Project",d.get("project_path","Standalone RMSNorm")),("Operator",display_operator(d.get("operator_id","rms_norm_train"))),("GPU",display_platform(d.get("platform_id","rtx5060_laptop_sm120"))),("Backend",display_backend(d.get("backend_id","cuda_cpp"))),("Campaign",d.get("campaign_id",DEFAULT_CAMPAIGN)),("Agent 工作目录",d.get("working_directory","尚未创建")),("状态",state)]);self.card("当前实验",[("Incumbent","RMSNorm V2"),("Episode",Path(d.get("working_directory","e0001")).parent.name if d.get("working_directory") else "Not started"),("Valid experiments",f"{disk.get('valid_experiment_count',disk.get('experiment_count',0))} / {disk.get('max_experiments',5)}"),("Framework attempts",str(disk.get('framework_attempt_count',0))),("Candidate",d.get("working_directory","None"))]);self.card("已有性能证据",[("Correctness","56 / 56 PASS"),("V2 vs V1","~1.074x arithmetic mean, repeated ABBA"),("Winning shapes","50 / 56"),("Resources","1056 B shared / block · 30 registers / thread")]);b=ttk.Frame(self.main);b.pack(fill="x",pady=10);ttk.Button(b,text="CHANGE WORKBENCH",command=lambda:WorkspaceManager(self)).pack(side="left");self.start_button=ttk.Button(b,text="START OPTIMIZATION",command=self.start,state="normal" if buttons["start_enabled"] else "disabled");self.start_button.pack(side="left",padx=6);ttk.Button(b,text="PAUSE AFTER STEP",command=self.pause,state="normal" if active else "disabled").pack(side="left");self.resume_button=ttk.Button(b,text="RESUME",command=self.resume,state="normal" if buttons["resume_enabled"] else "disabled");self.resume_button.pack(side="left",padx=4);ttk.Button(b,text="STOP SAFELY",command=self.stop,state="normal" if active else "disabled").pack(side="left",padx=6);ttk.Button(b,text="EMERGENCY STOP",command=self.emergency,state="normal" if active else "disabled").pack(side="left");ttk.Button(b,text="HARVEST RESULTS",command=self.harvest).pack(side="left")
    def refresh(self):
        if self.probe:
            workbench_state=self.controller.status().get("workbench",{}).get("workbench_state",self.probe.get("status"))
            # Environment readiness must not disguise an already-run episode as
            # a fresh PREPARED workbench after filesystem recovery.
            shown_state=self.episode_state if self.episode_state in RECOVERABLE_EPISODE_STATES else workbench_state
            self.banner.configure(text=f"LOCAL · {self.probe.get('gpu_name')} · sm_120 · {shown_state}\n{display_operator(self.active_draft.get('operator_id','rms_norm_train'))} · {display_backend(self.active_draft.get('backend_id','cuda_cpp'))} · GPT-5.6 Luna / Low · Environment {self.probe.get('environment_fingerprint','unknown')[:12]}...")
        if self.controller.status().get("runner_active"):
            self.after(1500,lambda:(self.overview(),self.refresh()))
    def start(self):
        d=self.active_draft;candidate=d.get("working_directory");campaign=d.get("campaign_id",DEFAULT_CAMPAIGN);incumbent=ROOT/"ops/rms_norm_v2"
        if not candidate or not Path(candidate).is_dir():
            messagebox.showwarning("需要已批准工作台","请先在 CHANGE WORKBENCH 中提出并批准 candidate 工作目录。",parent=self);return
        if not self.probe or self.probe.get("status")!="READY":
            messagebox.showerror("环境未就绪","本地 RTX5060 probe 未达到 READY，不能启动。",parent=self);return
        prompt=f"开始真实本地优化？\n\nGPU：{self.probe.get('gpu_name')} / sm_120\nOperator：{display_operator(d.get('operator_id','rms_norm_train'))}\nCampaign：{campaign}\n\nCandidate working directory：\n{candidate}\n\nIncumbent（只读）：\n{incumbent}\n\nAgent：GPT-5.6 Luna / Low\nMax experiments：5\n\n这将调用 Agent、修改 candidate、编译 CUDA 并运行 GPU correctness/benchmark。不会改写 ops/rms_norm_v2。"
        if not messagebox.askyesno("开始优化",prompt,parent=self):return
        try:self.controller.start_optimization(campaign_id=campaign,candidate_root=candidate,incumbent=incumbent,budget={"max_experiments":5,"max_consecutive_failures":3});self.refresh();self.overview()
        except Exception as exc:messagebox.showerror("无法启动",f"{type(exc).__name__}: {exc}",parent=self)
    def pause(self):
        try:self.controller.pause_optimization();messagebox.showinfo("暂停已请求","当前原子步骤结束后将暂停。",parent=self)
        except Exception as exc:messagebox.showerror("暂停失败",str(exc),parent=self)
    def stop(self):
        try:self.controller.stop_optimization();messagebox.showinfo("安全停止已请求","当前原子步骤结束后写 checkpoint 并停止。",parent=self)
        except Exception as exc:messagebox.showerror("停止失败",str(exc),parent=self)
    def resume(self):
        try:self.controller.resume_optimization(self.active_draft);self.refresh();self.overview()
        except Exception as exc:messagebox.showerror("继续失败",str(exc),parent=self)
    def emergency(self):
        if not messagebox.askyesno("紧急停止","紧急停止会立即终止当前 Agent / 编译 / benchmark 子进程。已完成记录会保留；当前未完成步骤不会被视为性能失败。是否继续？",parent=self):return
        try:self.controller.emergency_stop();self.overview();messagebox.showinfo("已中断","已写入 interrupted checkpoint。",parent=self)
        except Exception as exc:messagebox.showerror("紧急停止失败",str(exc),parent=self)
    def harvest(self):messagebox.showinfo("Harvest",self.controller.harvest(self.active_draft.get("campaign_id",DEFAULT_CAMPAIGN)))
    def help(self):
        manual=ROOT/"USER_MANUAL_CN.md"
        if manual.exists():TextWindow(self,"GPU Operator Lab 用户手册",manual.read_text(encoding="utf-8"))
        else:messagebox.showinfo("Help","用户手册尚未生成。")

def knowledge_smoke_test():
    errors=[];root=tk.Tk();root.withdraw();root.report_callback_exception=lambda exc,val,tb:errors.append(f"{exc.__name__}: {val}");browser=KnowledgeBrowser(root);browser.update_idletasks();browser.refresh_all();roots=[browser.knowledge_tree.item(n,"text") for n in browser.knowledge_tree.get_children()];assert roots,"knowledge tree has no roots";assert any("RMSNorm" in x for x in roots),roots
    for query in ("warp",chr(0x5F52)+chr(0x7EA6),"ABBA","BI-V150"):browser.search.set(query);browser.run_search();browser.update_idletasks();assert browser.search_tree.get_children(),f"query returned no results: {query}"
    browser.filters["operator"].set(display_operator("rms_norm_train"));browser.filters["platform"].set(display_platform("rtx5060_laptop_sm120"));browser.refresh_knowledge_tree();texts=[]
    def walk(n=""):
        for child in browser.knowledge_tree.get_children(n):texts.append(browser.knowledge_tree.item(child,"text"));walk(child)
    walk()
    for version in ("V0","V1","V2"):assert any(version in x for x in texts),f"missing {version}"
    browser.destroy();root.destroy();assert not errors,"Tk callback exceptions: "+"; ".join(errors)
    report="RUNTIME_GUI_SMOKE = PASS\nCALLBACK_EXCEPTIONS = 0\nTREE ROOTS: "+", ".join(roots)+"\nRMSNorm / RTX5060: V0, V1, V2 present\nQUERY warp / 归约 / ABBA / BI-V150 = PASS"
    print(report.encode("ascii","backslashreplace").decode("ascii"))
def console_smoke_test():
    """Tk runtime smoke without Agent, evaluator, or formal event writes."""
    errors=[];root=tk.Tk();root.withdraw();root.controller=LabController(ROOT);root.active_draft=read_json(LAB/"runtime/workspace_draft.json",{}) or {};root.report_callback_exception=lambda exc,val,tb:errors.append(f"{exc.__name__}: {val}")
    console=AgentConsole(root)
    # Directly replay the immutable real event store, including historical
    # e0005/e0006 legacy DIAGNOSTIC_STARTED payloads.  Do not rely on polling
    # timing for this compatibility assertion.
    real_events=console.ctrl.events.replay()
    for event in real_events:console._add_event(event)
    root.update();assert any(item.get("type")=="DIAGNOSTIC_STARTED" for item in console._stream_items.values()),"real diagnostic history missing"
    synthetic=[
        ("KNOWLEDGE_REFRESHED",{"knowledge_ids":["test-k"]}),
        ("HYPOTHESIS_RECORDED",{"hypothesis":{"claim":"test reduction"},"bottleneck":"test"}),
        ("CANDIDATE_UPDATED",{"summary":"candidate.py +1 -0"}),
        ("COMPILE_RESULT",{"summary":"PASS · TEST_ONLY"}),
        ("CORRECTNESS_RESULT",{"summary":"56 / 56 PASS"}),
        ("BENCHMARK_RESULT",{"summary":"Development ABBA = 1.000x"}),
        ("AGENT_DECISION",{"decision":"REJECT_AND_CONTINUE"}),
        ("KNOWLEDGE_CANDIDATE_CREATED",{"summary":"TEST_ONLY OBSERVATION"}),
    ]
    # A long stream proves the follow behaviour instead of merely proving the
    # widgets instantiate.  These objects remain in memory only.
    for i in range(100):
        typ,payload=synthetic[i%len(synthetic)]
        console._add_event({"event_id":f"test-console-{i}","timestamp":f"2026-09-16T10:42:{i%60:02d}Z","type":typ,"payload":payload})
    root.update();assert len(console._stream_items)>=100;assert console.stream.get_children();assert console.stream.yview()[1]>=.99,console.stream.yview()
    console._add_event({"event_id":"test-console-101","timestamp":"2026-09-16T10:43:00Z","type":"AGENT_STATUS","payload":{"status":"TEST_ONLY_LATEST"}})
    root.update();assert console.stream.yview()[1]>=.99,console.stream.yview()
    render_markdown(console.conversation,"# 标题\n\n**粗体**\n\n- item\n\n`inline code`\n\n```cpp\n__global__ void foo() {}\n```\n\n"+("最后一行。\n"*160))
    # Tk Text reports a tiny non-scrollable terminal newline on some Windows
    # builds (the upper fraction can be ~0.994 instead of exactly 1.0).
    # >= .99 means the final rendered content is visible, unlike the old
    # initial/top position.
    root.update();assert console.conversation.yview()[1]>=.99,console.conversation.yview()
    console.tabs.select(console.conversation_page);root.update();assert console.conversation.yview()[1]>=.99,console.conversation.yview()
    console.tabs.select(console.stream_page);root.update();assert console.stream.yview()[1]>=.99,console.stream.yview()
    console.destroy();console=AgentConsole(root);root.update();console._follow_latest();root.update();assert console.conversation.yview()[1]>=.99,console.conversation.yview();console.destroy();root.destroy();assert not errors,"Tk callback exceptions: "+"; ".join(errors);print("REAL_EVENT_REPLAY = PASS\nCONSOLE_ALWAYS_FOLLOW_LATEST_PASS\nSYNTHETIC_EVENT_COLORS = PASS\nMARKDOWN_RENDER = PASS\nCALLBACK_EXCEPTIONS = 0")
def workspace_smoke_test():
    """Exercise selection and legacy-campaign compatibility without writes."""
    errors=[];root=tk.Tk();root.withdraw();root.controller=LabController(ROOT);root.probe={};root.refresh=lambda:None
    root.report_callback_exception=lambda exc,val,tb:errors.append(f"{exc.__name__}: {val}")
    manager=WorkspaceManager(root);manager._save=lambda:None
    manager.draft.update(project_id="rms_norm_v2",project_path=r"C:\\Users\\38154\\projects\\aka-local\\ops\\rms_norm_v2",operator_id="rms_norm_train",platform_id="rtx5060_laptop_sm120",backend_id="cuda_cpp");manager._render_campaigns()
    assert len(manager.compatible_campaigns)==1,manager.compatible_campaigns
    assert manager.compatible_campaigns[0]["id"]=="rms_norm_train__rtx5060_sm120__cuda_cpp"
    assert len(manager.campaign_tree.get_children())==1
    manager.operator_var.set(display_operator("dense_fused_attention"));manager.use_operator();assert manager.draft["operator_id"]=="dense_fused_attention"
    manager.operator_var.set(display_operator("rms_norm_train"));manager.use_operator();assert manager.draft["operator_id"]=="rms_norm_train"
    manager.destroy();root.destroy();assert not errors,"Tk callback exceptions: "+"; ".join(errors)
    print("WORKSPACE_SWITCH_SMOKE = PASS\nCOMPATIBLE_CAMPAIGN_COUNT = 1\nCOMPATIBLE_CAMPAIGN = rms_norm_train__rtx5060_sm120__cuda_cpp\nRMSNorm -> Dense Fused Attention -> RMSNorm")
def main():
    if "--knowledge-smoke-test" in sys.argv:knowledge_smoke_test();return
    if "--console-smoke-test" in sys.argv:console_smoke_test();return
    if "--workspace-smoke-test" in sys.argv:workspace_smoke_test();return
    App().mainloop()
if __name__=="__main__":main()
