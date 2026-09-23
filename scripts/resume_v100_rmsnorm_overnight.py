#!/usr/bin/env python3
"""Crash-resumable controller for the V100 RMSNorm overnight campaign.

No credential is accepted on the command line or persisted.  Remote inspection
uses getpass and keeps the password only in the current Python process.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paramiko


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "overnight" / "v100_rmsnorm_e2e_20260922"
STATE_PATH = ROOT / "STATE.json"
JOURNAL_PATH = ROOT / "JOURNAL.jsonl"
RUN_INDEX_PATH = ROOT / "RUN_INDEX.json"
REMOTE_JOBS_PATH = ROOT / "REMOTE_JOBS.json"
CONTRACT_PATH = ROOT / "controlled_v100_rmsnorm_e2e_contract.json"
LOCK_PATH = ROOT / "CONTRACT_LOCK.json"
GIT_SAFE = "D:/Users/38154/Downloads/aka-local-publish"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        if os.name != "nt":
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def append_event(event: str, status: str, notes: str = "", **fields: Any) -> None:
    state = read_json(STATE_PATH, {})
    record = {
        "timestamp": utc_now(),
        "phase": state.get("current_phase"),
        "event": event,
        "candidate_hash": fields.pop("candidate_hash", None),
        "remote_job_id": fields.pop("remote_job_id", None),
        "status": status,
        "artifact_path": fields.pop("artifact_path", None),
        "notes": notes,
        **fields,
    }
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_read(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={GIT_SAFE}", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def validate_contract(state: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    if not LOCK_PATH.exists():
        findings.append("CONTRACT_LOCK_MISSING")
        return findings
    lock = read_json(LOCK_PATH)
    if not CONTRACT_PATH.exists():
        findings.append("CONTRACT_FILE_MISSING")
        return findings
    actual = sha256(CONTRACT_PATH)
    if actual != lock.get("contract_sha256"):
        findings.append("CONTRACT_HASH_MISMATCH")
    if state.get("contract_hash") not in (None, actual):
        findings.append("STATE_CONTRACT_HASH_MISMATCH")
    return findings


def connect_remote(host: str, user: str) -> paramiko.SSHClient:
    password = getpass.getpass(f"SSH password for {user}@{host}: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=15, banner_timeout=15)
    password = ""  # noqa: F841 - explicitly drop the only Python reference
    return client


def remote_exec(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=30)
    del stdin
    code = stdout.channel.recv_exit_status()
    return code, stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


def remote_put_atomic(client: paramiko.SSHClient, local: Path, remote: str) -> None:
    """Upload through an exec channel when the server has no SFTP subsystem."""
    tmp = remote + ".upload.tmp"
    command = f"umask 027; cat > {shlex.quote(tmp)}; sync {shlex.quote(tmp)}; mv {shlex.quote(tmp)} {shlex.quote(remote)}"
    stdin, stdout, stderr = client.exec_command(command, timeout=60)
    with local.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            stdin.channel.sendall(chunk)
    stdin.channel.shutdown_write()
    code = stdout.channel.recv_exit_status()
    error = stderr.read().decode("utf-8", "replace")
    if code:
        raise RuntimeError(f"remote upload failed for {local.name}: {error}")


def remote_get_atomic(client: paramiko.SSHClient, remote: str, local: Path) -> None:
    command = f"cat {shlex.quote(remote)}"
    stdin, stdout, stderr = client.exec_command(command, timeout=120)
    del stdin
    data = stdout.read()
    code = stdout.channel.recv_exit_status()
    error = stderr.read().decode("utf-8", "replace")
    if code:
        raise RuntimeError(f"remote download failed for {remote}: {error}")
    local.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=local.name + ".", suffix=".tmp", dir=local.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, local)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def check_remote(state: dict[str, Any], client: paramiko.SSHClient) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    workspace = state.get("remote_workspace")
    if not workspace:
        return [{"status": "REMOTE_WORKSPACE_NOT_SET"}]
    code, out, err = remote_exec(client, f"test -d {workspace!r}; printf '%s' $?")
    exists = code == 0 and out.strip().endswith("0")
    results.append({"workspace": workspace, "exists": exists, "stderr": err.strip()})
    jobs = read_json(REMOTE_JOBS_PATH, {"jobs": []}).get("jobs", [])
    for job in jobs:
        if job.get("status") not in {"QUEUED", "RUNNING"}:
            continue
        status_path = job.get("remote_status_path")
        if not status_path:
            continue
        code, out, err = remote_exec(client, f"test -f {status_path!r} && cat {status_path!r}")
        results.append({"job_id": job.get("job_id"), "code": code, "status_text": out, "stderr": err})
    return results


def upload_campaign(client: paramiko.SSHClient, state: dict[str, Any]) -> dict[str, Any]:
    workspace = state["remote_workspace"]
    code, _, err = remote_exec(client, f"mkdir -p {shlex.quote(workspace)}/package {shlex.quote(workspace)}/remote_jobs")
    if code:
        raise RuntimeError(f"remote mkdir failed: {err}")
    local_files = [
        CONTRACT_PATH,
        LOCK_PATH,
        ROOT / "agent_blind_package" / "authentic_torch_norm_excerpt.py",
        ROOT / "agent_blind_package" / "frozen_shapes.json",
        ROOT / "agent_blind_package" / "PerformanceFacts.json",
        ROOT / "agent_blind_package" / "excluded_sources.json",
        ROOT / "agent_blind_package" / "blind_package_manifest.json",
        ROOT / "agent_blind_package" / "package_hashes.json",
        ROOT / "agent_blind_package" / "reference_runner.py",
        ROOT / "agent_blind_package" / "run_reference_job.sh",
        ROOT / "agent_blind_package" / "candidate_runner.py",
        ROOT / "agent_blind_package" / "run_candidate_job.sh",
        ROOT / "agent_blind_package" / "l1_l2_runner.py",
        ROOT / "agent_blind_package" / "run_l1_l2_job.sh",
        ROOT / "hypotheses" / "H001" / "attempts" / "1" / "candidate.py",
    ]
    uploaded: dict[str, str] = {}
    for local in local_files:
        remote = f"{workspace}/package/{local.name}"
        remote_put_atomic(client, local, remote)
        uploaded[local.name] = sha256(local)
    for local, remote_name in [
        (ROOT / "hypotheses" / "H001" / "attempts" / "1" / "candidate.py", "candidate_base.py"),
        (ROOT / "hypotheses" / "H001" / "attempts" / "2" / "candidate.py", "candidate_attempt2.py"),
        (ROOT / "hypotheses" / "H002" / "attempts" / "1" / "candidate.py", "candidate_H002_attempt1.py"),
        (ROOT / "hypotheses" / "H003" / "attempts" / "1" / "candidate.py", "candidate_H003_attempt1.py"),
    ]:
        remote_put_atomic(client, local, f"{workspace}/package/{remote_name}")
        uploaded[remote_name] = sha256(local)
    code, _, err = remote_exec(client, f"chmod 750 {shlex.quote(workspace)}/package/run_reference_job.sh {shlex.quote(workspace)}/package/run_candidate_job.sh {shlex.quote(workspace)}/package/run_l1_l2_job.sh")
    if code:
        raise RuntimeError(f"remote chmod failed: {err}")
    code, out, err = remote_exec(client, f"find {shlex.quote(workspace)}/package -maxdepth 1 -type f -exec sha256sum {{}} \\;")
    if code:
        raise RuntimeError(f"remote hash failed: {err}")
    remote_hashes = {line.split()[1].rsplit("/", 1)[-1]: line.split()[0] for line in out.splitlines() if len(line.split()) >= 2}
    mismatches = [name for name, digest in uploaded.items() if remote_hashes.get(name) != digest]
    if mismatches:
        raise RuntimeError(f"remote upload hash mismatch: {mismatches}")
    return {"uploaded": uploaded, "remote_hashes": remote_hashes}


def launch_reference(client: paramiko.SSHClient, state: dict[str, Any]) -> dict[str, Any]:
    jobs_doc = read_json(REMOTE_JOBS_PATH, {"schema_version": 1, "jobs": []})
    active = [j for j in jobs_doc["jobs"] if j.get("status") in {"QUEUED", "RUNNING"}]
    if active:
        raise RuntimeError(f"refusing duplicate launch; active jobs: {[j['job_id'] for j in active]}")
    job_id = "reference_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace = state["remote_workspace"]
    job_dir = f"{workspace}/remote_jobs/{job_id}"
    wrapper = f"{workspace}/package/run_reference_job.sh"
    command = (
        f"mkdir -p {shlex.quote(job_dir)}; "
        f"nohup {shlex.quote(wrapper)} {shlex.quote(workspace)} {shlex.quote(job_id)} "
        f">{shlex.quote(job_dir + '/launcher.stdout')} 2>{shlex.quote(job_dir + '/launcher.stderr')} "
        f"</dev/null & echo $!"
    )
    code, out, err = remote_exec(client, command)
    if code:
        raise RuntimeError(f"reference launch failed: {err}")
    pid = int(out.strip().splitlines()[-1])
    record = {
        "job_id": job_id,
        "kind": "REFERENCE_L0_L1",
        "pid": pid,
        "status": "RUNNING",
        "remote_job_dir": job_dir,
        "remote_status_path": f"{job_dir}/status.json",
        "remote_result_path": f"{job_dir}/result.json",
        "created_at": utc_now(),
        "contract_hash": state["contract_hash"],
        "candidate_hash": "AUTHENTIC_TORCH_REFERENCE",
        "gpu_uuid": state["selected_gpu_uuid"],
    }
    jobs_doc["jobs"].append(record)
    atomic_json(REMOTE_JOBS_PATH, jobs_doc)
    state["active_remote_job"] = job_id
    state["current_phase"] = "PHASE_B_REFERENCE"
    state["current_substep"] = "reference_l0_l1_running"
    state["next_action"] = "poll active reference job; ingest result when DONE; do not launch duplicate"
    atomic_json(STATE_PATH, state)
    append_event("REMOTE_JOB_LAUNCHED", "RUNNING", remote_job_id=job_id, notes=f"pid={pid}")
    return record


def launch_candidate(client: paramiko.SSHClient, state: dict[str, Any], mode: str, hypothesis: str, attempt: int) -> dict[str, Any]:
    jobs_doc = read_json(REMOTE_JOBS_PATH, {"schema_version": 1, "jobs": []})
    active = [j for j in jobs_doc["jobs"] if j.get("status") in {"QUEUED", "RUNNING"}]
    if active:
        raise RuntimeError(f"refusing duplicate launch; active jobs: {[j['job_id'] for j in active]}")
    candidate = ROOT / "hypotheses" / hypothesis / "attempts" / str(attempt) / "candidate.py"
    candidate_hash = sha256(candidate)
    job_id = f"{mode}_{hypothesis}_a{attempt}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace = state["remote_workspace"]
    job_dir = f"{workspace}/remote_jobs/{job_id}"
    wrapper = f"{workspace}/package/run_candidate_job.sh"
    if hypothesis == "H001":
        remote_candidate = f"{workspace}/package/candidate.py" if attempt == 1 else f"{workspace}/package/candidate_attempt2.py"
    else:
        remote_candidate = f"{workspace}/package/candidate_{hypothesis}_attempt{attempt}.py"
    command = (
        f"mkdir -p {shlex.quote(job_dir)}; "
        f"nohup {shlex.quote(wrapper)} {shlex.quote(workspace)} {shlex.quote(job_id)} "
        f"{shlex.quote(remote_candidate)} {shlex.quote(mode)} >{shlex.quote(job_dir + '/launcher.stdout')} "
        f"2>{shlex.quote(job_dir + '/launcher.stderr')} </dev/null & echo $!"
    )
    code, out, err = remote_exec(client, command)
    if code:
        raise RuntimeError(f"candidate launch failed: {err}")
    pid = int(out.strip().splitlines()[-1])
    record = {
        "job_id": job_id, "kind": "QUICK_L0" if mode == "quick" else "OFFICIAL_L0", "pid": pid, "status": "RUNNING",
        "remote_job_dir": job_dir, "remote_status_path": f"{job_dir}/status.json",
        "remote_result_path": f"{job_dir}/result.json", "created_at": utc_now(),
        "contract_hash": state["contract_hash"], "candidate_hash": candidate_hash,
        "gpu_uuid": state["selected_gpu_uuid"], "hypothesis_id": hypothesis, "attempt_id": attempt,
    }
    jobs_doc["jobs"].append(record)
    atomic_json(REMOTE_JOBS_PATH, jobs_doc)
    state["active_remote_job"] = job_id
    state["active_hypothesis_id"] = hypothesis
    state["active_attempt_id"] = attempt
    state["current_phase"] = "PHASE_D_AGENT_SEARCH"
    state["current_substep"] = f"{hypothesis}_attempt{attempt}_{mode}_l0_running"
    state["next_action"] = f"poll {hypothesis} attempt {attempt} {mode.upper()} L0; ingest and classify before promotion"
    atomic_json(STATE_PATH, state)
    append_event("REMOTE_JOB_LAUNCHED", "RUNNING", remote_job_id=job_id,
                 candidate_hash=candidate_hash, notes=f"{hypothesis} attempt {attempt} {mode.upper()} L0 pid={pid}")
    return record


def launch_l1_l2(client: paramiko.SSHClient, state: dict[str, Any], mode: str, hypothesis: str, attempt: int) -> dict[str, Any]:
    jobs_doc = read_json(REMOTE_JOBS_PATH, {"schema_version": 1, "jobs": []})
    active = [j for j in jobs_doc["jobs"] if j.get("status") in {"QUEUED", "RUNNING"}]
    if active:
        raise RuntimeError(f"refusing duplicate launch; active jobs: {[j['job_id'] for j in active]}")
    candidate = ROOT / "hypotheses" / hypothesis / "attempts" / str(attempt) / "candidate.py"
    candidate_hash = sha256(candidate)
    job_id = f"{mode}_{hypothesis}_a{attempt}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace = state["remote_workspace"]
    job_dir = f"{workspace}/remote_jobs/{job_id}"
    wrapper = f"{workspace}/package/run_l1_l2_job.sh"
    if hypothesis == "H001":
        remote_candidate = f"{workspace}/package/candidate.py" if attempt == 1 else f"{workspace}/package/candidate_attempt2.py"
    else:
        remote_candidate = f"{workspace}/package/candidate_{hypothesis}_attempt{attempt}.py"
    command = (
        f"mkdir -p {shlex.quote(job_dir)}; nohup {shlex.quote(wrapper)} {shlex.quote(workspace)} "
        f"{shlex.quote(job_id)} {shlex.quote(remote_candidate)} {shlex.quote(mode)} "
        f">{shlex.quote(job_dir + '/launcher.stdout')} 2>{shlex.quote(job_dir + '/launcher.stderr')} "
        f"</dev/null & echo $!"
    )
    code, out, err = remote_exec(client, command)
    if code:
        raise RuntimeError(f"L1/L2 launch failed: {err}")
    pid = int(out.strip().splitlines()[-1])
    record = {
        "job_id": job_id, "kind": "L1_SMOKE" if mode == "smoke" else "L2_CONTROLLED_MEGATRON_E2E",
        "pid": pid, "status": "RUNNING", "remote_job_dir": job_dir,
        "remote_status_path": f"{job_dir}/status.json", "remote_result_path": f"{job_dir}/result.json",
        "created_at": utc_now(), "contract_hash": state["contract_hash"],
        "candidate_hash": candidate_hash, "gpu_uuid": state["selected_gpu_uuid"],
        "hypothesis_id": hypothesis, "attempt_id": attempt,
    }
    jobs_doc["jobs"].append(record)
    atomic_json(REMOTE_JOBS_PATH, jobs_doc)
    state["active_remote_job"] = job_id
    state["current_phase"] = "PHASE_E_L1_L2"
    state["current_substep"] = f"{hypothesis}_attempt{attempt}_{mode}_running"
    state["next_action"] = f"poll {job_id}; ingest and validate real Megatron replacement before promotion"
    atomic_json(STATE_PATH, state)
    append_event("REMOTE_JOB_LAUNCHED", "RUNNING", remote_job_id=job_id,
                 candidate_hash=candidate_hash, notes=f"{hypothesis} attempt {attempt} L1/L2 {mode} pid={pid}")
    return record


def ingest_jobs(client: paramiko.SSHClient, state: dict[str, Any]) -> list[dict[str, Any]]:
    jobs_doc = read_json(REMOTE_JOBS_PATH, {"schema_version": 1, "jobs": []})
    ingested: list[dict[str, Any]] = []
    changed = False
    for job in jobs_doc["jobs"]:
        if job.get("status") not in {"QUEUED", "RUNNING"}:
            continue
        code, out, err = remote_exec(client, f"cat {shlex.quote(job['remote_status_path'])}")
        if code:
            ingested.append({"job_id": job["job_id"], "status": "STATUS_UNAVAILABLE", "stderr": err})
            continue
        remote_status = json.loads(out)
        status = remote_status.get("status")
        if status in {"QUEUED", "RUNNING"}:
            ingested.append({"job_id": job["job_id"], "status": status})
            continue
        job["status"] = status
        job["finished_at"] = utc_now()
        local_dir = ROOT / "remote_artifacts" / job["job_id"]
        code, names, err = remote_exec(
            client,
            f"find {shlex.quote(job['remote_job_dir'])} -maxdepth 1 -type f -printf '%f\\n'",
        )
        if code:
            raise RuntimeError(f"remote artifact listing failed: {err}")
        downloads = {}
        for name in [n for n in names.splitlines() if n and "/" not in n and "\\" not in n]:
            local = local_dir / name
            remote_get_atomic(client, f"{job['remote_job_dir']}/{name}", local)
            downloads[name] = sha256(local)
        job["local_artifact_dir"] = str(local_dir.relative_to(REPO)).replace("\\", "/")
        job["download_sha256"] = downloads
        if job.get("kind") == "ENVIRONMENT_SETUP" and status == "DONE":
            state["current_substep"] = "compatible_environment_ready"
            state["next_action"] = "launch replacement reference L0/L1 job with isolated torch 2.7.1+cu118 environment"
        elif status == "DONE" and (local_dir / "result.json").exists():
            result = read_json(local_dir / "result.json")
            key = "|".join([
                job["contract_hash"], job["candidate_hash"], job["kind"],
                "ALL_OFFICIAL_SHAPES", "v100_rmsnorm_e2e_v1",
            ])
            run_index = read_json(RUN_INDEX_PATH)
            run_index["runs"][key] = {
                "status": "DONE", "job_id": job["job_id"], "result": job["local_artifact_dir"] + "/result.json",
                "sha256": downloads["result.json"],
            }
            atomic_json(RUN_INDEX_PATH, run_index)
            state["completed_runs"] = state.get("completed_runs", 0) + 1
        else:
            state["failed_runs"] = state.get("failed_runs", 0) + 1
        if state.get("active_remote_job") == job["job_id"]:
            state["active_remote_job"] = None
        append_event("REMOTE_JOB_INGESTED", status, remote_job_id=job["job_id"],
                     artifact_path=job["local_artifact_dir"], notes=remote_status.get("detail", ""))
        ingested.append({"job_id": job["job_id"], "status": status, "downloads": downloads})
        changed = True
    if changed:
        atomic_json(REMOTE_JOBS_PATH, jobs_doc)
        atomic_json(STATE_PATH, state)
    return ingested


def resume(args: argparse.Namespace) -> int:
    if not STATE_PATH.exists():
        print(f"BLOCKED: missing {STATE_PATH}")
        return 2
    state = read_json(STATE_PATH)
    findings: list[str] = []

    status = git_read("status", "--short", "--branch")
    branch = git_read("branch", "--show-current").stdout.strip()
    if branch != "overnight/v100-rmsnorm-e2e-20260922":
        findings.append("CHALLENGE_BRANCH_NOT_ACTIVE")
    if status.returncode != 0:
        findings.append("GIT_STATUS_FAILED")
    findings.extend(validate_contract(state))

    for required in (REMOTE_JOBS_PATH, RUN_INDEX_PATH):
        try:
            read_json(required)
        except Exception as exc:  # preserve evidence and continue other checks
            findings.append(f"INVALID_JSON:{required.name}:{exc}")

    remote_results: list[dict[str, Any]] = []
    if args.remote or args.upload_blind or args.launch_reference or args.launch_candidate or args.launch_l1l2 or args.ingest:
        client = connect_remote(args.host, args.user)
        try:
            remote_results = check_remote(state, client)
            if args.upload_blind or args.launch_reference or args.launch_candidate or args.launch_l1l2:
                upload_result = upload_campaign(client, state)
                remote_results.append({"upload": upload_result})
                append_event("BLIND_PACKAGE_UPLOADED", "PASS", notes="local and remote SHA256 match")
            if args.launch_reference:
                remote_results.append({"launched": launch_reference(client, state)})
            if args.launch_candidate:
                remote_results.append({"launched": launch_candidate(client, state, args.candidate_mode, args.hypothesis, args.attempt)})
            if args.launch_l1l2:
                remote_results.append({"launched": launch_l1_l2(client, state, args.l1l2_mode, args.hypothesis, args.attempt)})
            if args.ingest:
                remote_results.append({"ingested": ingest_jobs(client, state)})
        finally:
            client.close()

    state["last_checkpoint_time"] = utc_now()
    state["resume_findings"] = findings
    state["git_status_snapshot"] = status.stdout.strip()
    if remote_results:
        state["last_remote_resume_check"] = remote_results
    atomic_json(STATE_PATH, state)
    append_event("RESUME_CHECK", "PASS" if not findings else "PARTIAL", notes=";".join(findings))
    print(json.dumps({
        "campaign_id": state.get("campaign_id"),
        "current_phase": state.get("current_phase"),
        "current_substep": state.get("current_substep"),
        "next_action": state.get("next_action"),
        "deadline_utc": state.get("deadline_utc"),
        "findings": findings,
        "remote": remote_results,
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="prompt for SSH password and inspect remote jobs")
    parser.add_argument("--upload-blind", action="store_true", help="upload the hash-locked blind package")
    parser.add_argument("--launch-reference", action="store_true", help="upload and launch the persistent reference job")
    parser.add_argument("--launch-candidate", action="store_true", help="upload and launch H001 attempt 1 QUICK L0")
    parser.add_argument("--candidate-mode", choices=["quick", "official"], default="quick")
    parser.add_argument("--attempt", type=int, choices=[1, 2], default=1)
    parser.add_argument("--hypothesis", choices=["H001", "H002", "H003"], default="H001")
    parser.add_argument("--launch-l1l2", action="store_true", help="upload and launch H001 L1/L2 evaluator")
    parser.add_argument("--l1l2-mode", choices=["smoke", "official"], default="smoke")
    parser.add_argument("--ingest", action="store_true", help="download and index terminal remote jobs")
    parser.add_argument("--host", default="10.130.147.227")
    parser.add_argument("--user", default="bencheng")
    args = parser.parse_args()
    return resume(args)


if __name__ == "__main__":
    raise SystemExit(main())
