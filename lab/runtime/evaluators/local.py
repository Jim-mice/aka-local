"""Real RTX5060 RMSNorm adapter using the established benchmark scripts."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
from .diagnostics import summarize_repeated_per_shape, write_repeated_artifacts
from .ncu_profile import BASIC_DIAGNOSTIC_METRICS, parsed_profile, write_profile_summary


class RTX5060LocalEvaluator:
    def __init__(self, root: str | Path, *, candidate: str | Path, incumbent: str | Path, evidence_dir: str | Path, dry_run=False, process_registry=None):
        self.root = Path(root).resolve()
        self.candidate = Path(candidate).resolve()
        self.incumbent = Path(incumbent).resolve()
        self.evidence_dir = Path(evidence_dir).resolve()
        self.dry_run = dry_run
        self.python = self.root / ".venv" / "Scripts" / "python.exe"
        self.abba = self.root / "benchmarks" / "rmsnorm_abba.py"
        self.repeated = self.root / "benchmarks" / "rmsnorm_repeated.py"
        self.shape_metadata = self.root.parent / "atrex-bench" / "data" / "rms_norm" / "shapes.json"
        self.ncu = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.0\target\windows-desktop-win7-x64\ncu.exe")
        self.vcvars = Path(r"D:\Microsoft C++ Build Tools\VC\Auxiliary\Build\vcvars64.bat")
        self.process_registry=process_registry

    @staticmethod
    def _implementation(path: Path) -> Path:
        """Legacy RMSNorm CLIs import a Python implementation file, not a directory."""
        path = Path(path)
        return path / "candidate.py" if path.is_dir() else path

    def _build_env(self):
        """Reuse the established Windows CUDA/MSVC order for every worker."""
        env = os.environ.copy()
        env["PATH"] = str(self.python.parent) + os.pathsep + env.get("PATH", "")
        cuda = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4")
        if cuda.exists():
            env["CUDA_PATH"] = str(cuda)
            env["PATH"] = str(cuda / "bin") + os.pathsep + env["PATH"]
        env["TORCH_CUDA_ARCH_LIST"] = "12.0"
        if self.vcvars.exists():
            command = f'call "{self.vcvars}" >nul && set'
            exported = subprocess.run(["cmd.exe", "/d", "/s", "/c", command], capture_output=True, text=True, encoding="utf-8", errors="replace")
            if exported.returncode == 0:
                for line in exported.stdout.splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        env[key] = value
        return env

    def _run(self, phase, command, output=None, *, timeout=None, on_tick=None):
        started = time.monotonic()
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            return {"phase": phase, "pass": True, "dry_run": True, "command": command, "duration_s": 0.0}
        if self.process_registry:
            from ..process_registry import ManagedProcessLauncher
            code,stdout,stderr=ManagedProcessLauncher(self.process_registry).run(command,kind=phase,cwd=self.root,env=self._build_env(),timeout=timeout,on_tick=on_tick)
        else:
            try: completed = subprocess.run(command, cwd=self.root, env=self._build_env(), capture_output=True, text=True, encoding="utf-8", errors="replace",timeout=timeout);code,stdout,stderr=completed.returncode,completed.stdout,completed.stderr
            except subprocess.TimeoutExpired as exc: code,stdout,stderr=-1,exc.stdout or "",(exc.stderr or "")+"\nPROFILE_TIMEOUT"
        result = {"phase": phase, "pass": code == 0, "returncode": code, "stdout": stdout, "stderr": stderr, "duration_s": round(time.monotonic()-started, 3), "command": command}
        if output and Path(output).exists():
            try: result["metrics"] = json.loads(Path(output).read_text(encoding="utf-8"))
            except json.JSONDecodeError: result["metrics_parse_error"] = True
        (self.evidence_dir / f"{phase}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
        return result

    def prepare(self):
        missing = [str(path) for path in (self.python, self.abba, self.repeated, self.candidate, self.incumbent) if not path.exists()]
        return {"phase": "prepare", "pass": not missing, "missing": missing, "dry_run": self.dry_run}

    def compile(self):
        # Importing a candidate calls its existing load_inline build path. The
        # correctness worker gives a machine-readable result and is the source
        # of truth for compile success/failure.
        return self.check_correctness(phase="compile")

    def check_correctness(self, phase="correctness"):
        output = self.evidence_dir / f"{phase}_result.json"
        command = [str(self.python), "-m", "lab.runtime.evaluators.rmsnorm_correctness", "--candidate", str(self._implementation(self.candidate)), "--reference", str(self.root.parent / "atrex-bench" / "data" / "rms_norm" / "reference.py"), "--output", str(output)]
        result = self._run(phase, command, output)
        metrics = result.get("metrics", {})
        result["pass"] = bool(result.get("pass") and metrics.get("pass"))
        return result

    def development_benchmark(self):
        output = self.evidence_dir / "development_abba.json"
        command = [str(self.python), str(self.abba), "--candidate", str(self._implementation(self.candidate)), "--baseline", str(self._implementation(self.incumbent)), "--output", str(output), "--warmup", "2", "--repeats", "5"]
        result = self._run("development_benchmark", command, output)
        result["measurement_method"] = "DEVELOPMENT same-process A/B/B/A, warmup=2, repeats=5"
        return result

    def authoritative_abba(self):
        output = self.evidence_dir / "authoritative_abba.json"
        command = [str(self.python), str(self.abba), "--candidate", str(self._implementation(self.candidate)), "--baseline", str(self._implementation(self.incumbent)), "--output", str(output), "--warmup", "20", "--repeats", "100"]
        result = self._run("authoritative_abba", command, output)
        result["measurement_method"] = "AUTHORITATIVE same-process A/B/B/A, warmup=20, repeats=100"
        return result

    def robustness(self):
        output = self.evidence_dir / "robustness_abba.json"
        command = [str(self.python), str(self.repeated), "--incumbent", str(self._implementation(self.incumbent)), "--candidate", str(self._implementation(self.candidate)), "--output", str(output), "--batches", "5", "--warmup", "20", "--repeats", "100"]
        result = self._run("robustness", command, output)
        result["measurement_method"] = "ROBUSTNESS five same-process A/B/B/A batches"
        return result

    def benchmark(self): return self.development_benchmark()
    def inspect_shape_metadata(self):
        """Read the official shape inventory only; this never launches CUDA."""
        try:
            shapes = json.loads(self.shape_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"phase": "inspect_shape_metadata", "pass": False, "reason": f"SHAPE_METADATA_UNAVAILABLE: {exc}"}
        rows = []
        for shape_id, spec in sorted(shapes.items(), key=lambda item: int(item[0])):
            args = spec.get("input_kwargs") or {}
            rows.append({"shape_id": str(shape_id), "token_count": args.get("token_count"), "hidden_size": args.get("hidden_size"), "input_kwargs": args, "init_kwargs": spec.get("init_kwargs") or {}})
        return {"phase": "inspect_shape_metadata", "pass": True, "shape_count": len(rows), "shapes": rows, "source": str(self.shape_metadata), "source_sha256": hashlib.sha256(self.shape_metadata.read_bytes()).hexdigest()}

    def _diagnostic_policy(self):
        path = self.evidence_dir.parents[2] / "diagnostic_policy.json" if len(self.evidence_dir.parents) >= 3 else None
        if path and path.exists():
            try: return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError): pass
        return {"stability": {"max_cv": .05, "min_direction_consistency": .80, "min_effect_size": .01}, "scope_note": "campaign diagnostic defaults; not portable hardware rules"}

    def repeated_per_shape_benchmark(self, *, shapes="all", batches=5, warmup=20, repeats=100, progress_callback=None):
        """Run the existing repeated same-process ABBA script and materialize distributions.

        This is invoked only by an explicit DIAGNOSTIC action, never by the
        normal optimization gate.
        """
        if shapes != "all":
            return {"phase": "repeated_per_shape_benchmark", "pass": False, "reason": "SHAPE_SUBSET_NOT_SUPPORTED_BY_LEGACY_REPEATED_SCRIPT", "requested_shapes": shapes}
        root = self.evidence_dir / "repeated_per_shape"
        output = root / "legacy_repeated_output.json"
        progress_path=root/"progress.jsonl";seen=0
        def tick():
            nonlocal seen
            if not progress_callback or not progress_path.exists():return
            lines=progress_path.read_text(encoding="utf-8",errors="replace").splitlines()
            for line in lines[seen:]:
                try:progress_callback(json.loads(line))
                except json.JSONDecodeError:continue
            seen=len(lines)
        command = [str(self.python), str(self.repeated), "--incumbent", str(self._implementation(self.incumbent)), "--candidate", str(self._implementation(self.candidate)), "--output", str(output), "--batches", str(int(batches)), "--warmup", str(int(warmup)), "--repeats", str(int(repeats)), "--progress-jsonl", str(progress_path)]
        result = self._run("repeated_per_shape_benchmark", command, output,on_tick=tick)
        tick()
        raw = result.get("metrics") or {}
        if not result.get("pass"):
            return result
        raw["batches"], raw["warmup"], raw["repeats"] = int(batches), int(warmup), int(repeats)
        identical = self._tree_hash(self.candidate) == self._tree_hash(self.incumbent)
        summary = summarize_repeated_per_shape(raw, identical_implementation=identical, policy=self._diagnostic_policy())
        artifacts = write_repeated_artifacts(root, raw, summary)
        result.update({"metrics": summary, "raw_measurements": artifacts["raw"], "artifacts": artifacts, "identical_implementation": identical})
        return result

    @staticmethod
    def _tree_hash(root):
        digest = hashlib.sha256(); root = Path(root)
        for path in sorted(root.rglob("*")):
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
                digest.update(str(path.relative_to(root)).encode("utf-8")); digest.update(path.read_bytes())
        return digest.hexdigest()

    def static_evidence(self):
        """Collect source/static facts without compiling or profiling."""
        candidate_file = self._implementation(self.candidate)
        evidence = {"phase": "static_evidence", "pass": candidate_file.exists(), "candidate": str(candidate_file), "candidate_sha256": hashlib.sha256(candidate_file.read_bytes()).hexdigest() if candidate_file.exists() else None,
                    "compile_architecture": "sm_120", "threads_per_block": None, "grid_policy": None, "registers_per_thread": None, "shared_memory_per_block": None, "spills": None, "stack_or_local": None, "kernel_symbols": [], "evidence_sources": []}
        # First use the incumbent/candidate's recorded static manifest when it
        # exists; it is provenance, not an inferred compiler result.
        for manifest in (self.candidate / "manifest.json", self.incumbent / "manifest.json"):
            try: item=json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError): continue
            static=item.get("static") or {}
            evidence["evidence_sources"].append(str(manifest))
            evidence["registers_per_thread"]=static.get("registers_per_thread",evidence["registers_per_thread"])
            evidence["shared_memory_per_block"]=static.get("shared_bytes_per_block",evidence["shared_memory_per_block"])
            evidence["stack_or_local"]={"stack_bytes":static.get("stack_bytes"),"local_bytes":static.get("local_bytes")} if static else evidence["stack_or_local"]
            break
        # Reuse real compiler/evaluator artifacts if they already exist.  The
        # parser intentionally refuses to invent unavailable fields.
        for path in sorted(self.evidence_dir.glob("compile*.json")):
            try: item=json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError): continue
            text="\n".join(str(item.get(key,"")) for key in ("stdout","stderr"))
            evidence["evidence_sources"].append(str(path))
            import re
            reg=re.search(r"(?:Used|uses)\s+(\d+)\s+register",text,re.I); shared=re.search(r"(?:shared memory|smem)[^\d]*(\d+)\s*(?:bytes|B)",text,re.I)
            if reg: evidence["registers_per_thread"]=int(reg.group(1))
            if shared: evidence["shared_memory_per_block"]=int(shared.group(1))
        out=self.evidence_dir/"static_evidence.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");evidence["artifact"]=str(out)
        return evidence

    def profiler_capabilities(self):
        """Discover only NCU availability/version; never starts a profile."""
        if not self.ncu.exists(): return {"phase":"profiler_capabilities","pass":False,"status":"CAPABILITY_MISSING","reason":"NCU_NOT_FOUND","path":str(self.ncu)}
        completed=subprocess.run([str(self.ncu),"--version"],capture_output=True,text=True,encoding="utf-8",errors="replace")
        return {"phase":"profiler_capabilities","pass":completed.returncode==0,"status":"AVAILABLE" if completed.returncode==0 else "CAPABILITY_MISSING","path":str(self.ncu),"version":completed.stdout.strip(),"stderr":completed.stderr.strip(),"profile_not_started":True}

    def profile(self, *, shape_ids=None, questions=None, test_only=False):
        caps=self.profiler_capabilities()
        if not caps.get("pass"): return caps
        ids=[int(item) for item in (shape_ids or [])]
        if not ids: return {**caps,"pass":False,"status":"DIAGNOSTIC_ACTION_FAILED","reason":"PROFILE_SHAPES_REQUIRED"}
        if len(ids)>6:return {**caps,"pass":False,"status":"DIAGNOSTIC_ACTION_FAILED","reason":"PROFILE_REQUEST_TOO_LARGE","max_shapes":6}
        static=self.static_evidence();results=[]
        for shape_id in ids:
            destination=self.evidence_dir/"profiler"/f"shape_{shape_id}";destination.mkdir(parents=True,exist_ok=True)
            request={"type":"profiler","shape_id":shape_id,"questions":questions or [],"measurement":"PROFILE_MEASUREMENT","not_latency_benchmark":True,"test_only":bool(test_only),"kernel_filter":"regex:.*rmsnorm_row_kernel.*","launch_skip":20,"launch_count":1,"metrics":list(BASIC_DIAGNOSTIC_METRICS)}
            (destination/"profile_request.json").write_text(json.dumps(request,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            worker_metadata=destination/"worker_metadata.json";export_base=destination/"profile"
            command=[str(self.ncu),"--csv","--page","raw","--metrics",",".join(BASIC_DIAGNOSTIC_METRICS),"--kernel-name","regex:.*rmsnorm_row_kernel.*","--launch-skip","20","--launch-count","1","--export",str(export_base),"--force-overwrite",str(self.python),"-m","lab.runtime.evaluators.rmsnorm_profile_worker","--candidate",str(self._implementation(self.candidate)),"--shape-id",str(shape_id),"--warmup","20","--iterations","2","--metadata-output",str(worker_metadata)]
            (destination/"ncu_command.json").write_text(json.dumps({"command":command,"redacted":False},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            result=self._run(f"ncu_profile_shape_{shape_id}",command,timeout=300)
            (destination/"ncu_stdout.txt").write_text(result.get("stdout","") or "",encoding="utf-8");(destination/"ncu_stderr.txt").write_text(result.get("stderr","") or "",encoding="utf-8")
            if not worker_metadata.exists():
                results.append({"shape_id":shape_id,"pass":False,"status":"DIAGNOSTIC_ACTION_FAILED","reason":"PROFILE_WORKER_METADATA_MISSING","artifacts":{"directory":str(destination)}});continue
            worker=json.loads(worker_metadata.read_text(encoding="utf-8"));parsed=parsed_profile(worker,result.get("stdout","") or "",static_evidence=static)
            (destination/"metrics_raw.csv").write_text(result.get("stdout","") or "",encoding="utf-8")
            (destination/"metrics_parsed.json").write_text(json.dumps(parsed,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            write_profile_summary(destination/"profile_summary.md",parsed,static_source=static.get("artifact"))
            report=next((path for path in (export_base.with_suffix(".ncu-rep"),export_base.with_suffix(".ncu-repz")) if path.exists()),None)
            captured=bool(parsed.get("kernel",{}).get("name") and "rmsnorm_row_kernel" in str(parsed["kernel"]["name"]))
            item={"shape_id":shape_id,"pass":bool(result.get("pass") and captured),"status":"PROFILE_CAPTURED" if result.get("pass") and captured else "DIAGNOSTIC_ACTION_FAILED","kernel_captured":captured,"parsed":parsed,"artifacts":{"directory":str(destination),"profile_rep":str(report) if report else None,"worker_metadata":str(worker_metadata),"raw_csv":str(destination/"metrics_raw.csv"),"parsed_metrics":str(destination/"metrics_parsed.json"),"summary":str(destination/"profile_summary.md")},"ncu_returncode":result.get("returncode")}
            if not item["pass"]:item["reason"]="KERNEL_NOT_MATCHED" if result.get("pass") else "NCU_COMMAND_FAILED"
            results.append(item)
        return {**caps,"phase":"profile","pass":all(item.get("pass") for item in results),"status":"PROFILE_CAPTURED" if all(item.get("pass") for item in results) else "DIAGNOSTIC_ACTION_FAILED","questions":questions or [],"profiles":results,"measurement":"PROFILE_MEASUREMENT","not_latency_benchmark":True,"test_only":bool(test_only)}

    def compare_regimes(self, *, repeated_summary=None):
        if not repeated_summary:
            return {"phase":"compare_regimes","pass":False,"reason":"REPEATED_PER_SHAPE_EVIDENCE_REQUIRED"}
        return {"phase":"compare_regimes","pass":True,"regimes":repeated_summary.get("regimes",{}),"source_kind":repeated_summary.get("kind")}

    def run_diagnostic_action(self, action, progress_callback=None):
        action=dict(action or {}); kind=action.get("type"); params={key:value for key,value in action.items() if key!="type"}
        if kind=="inspect_shape_metadata": return self.inspect_shape_metadata()
        if kind=="repeated_per_shape_benchmark": return self.repeated_per_shape_benchmark(**params,progress_callback=progress_callback)
        if kind=="static_evidence": return self.static_evidence()
        if kind=="profiler":
            if params.get("capability_only"): return self.profiler_capabilities()
            return self.profile(shape_ids=params.get("shape_ids"),questions=params.get("questions"))
        if kind=="compare_regimes": return self.compare_regimes(repeated_summary=params.get("repeated_summary"))
        return {"phase":"diagnostic","pass":False,"reason":"UNKNOWN_DIAGNOSTIC_ACTION","action":kind}
    def cleanup(self): return {"phase": "cleanup", "pass": True}
