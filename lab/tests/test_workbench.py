import json
from pathlib import Path
from lab.core.credentials import EphemeralCredentialStore
from lab.core.events import EventStore
from lab.core.workbench import Workbench, environment_fingerprint
from lab.core.controller import LabController

def test_fingerprint_stable_and_changes():
    x={"platform_id":"rtx5060_laptop_sm120","gpu_device":"RTX 5060","gpu_index":0,"backend":"cuda_cpp"}
    assert environment_fingerprint(x)==environment_fingerprint(dict(x))
    y=dict(x); y["gpu_index"]=1; assert environment_fingerprint(x)!=environment_fingerprint(y)

def test_credentials_not_serialized(tmp_path):
    s=EphemeralCredentialStore(); ref=s.put("ssh", "secret-value"); assert s.get(ref)=="secret-value"
    assert "secret-value" not in json.dumps({"credential_ref":ref})

def test_event_idempotence_and_redaction(tmp_path):
    e=EventStore(tmp_path/"events.jsonl"); e.append("HUMAN_MESSAGE",{"password":"secret"},event_id="x"); e.append("HUMAN_MESSAGE",{"password":"secret"},event_id="x")
    assert len(e.replay())==1; assert "secret" not in (tmp_path/"events.jsonl").read_text()

def test_controller_refresh_requires_event(tmp_path):
    c=LabController(tmp_path); c.configure_local("rms_norm_train"); c.refresh_knowledge({"operator":"rms_norm_train","platform":"rtx5060_laptop_sm120"})
    assert any(x["type"]=="KNOWLEDGE_REFRESHED" for x in c.events.replay())

def test_remote_probe_is_planning_only():
    from lab.runtime.executors.remote import SSHExecutor
    assert SSHExecutor().probe()["status"]=="PLANNED"
