import tempfile
from pathlib import Path
from lab.core.handoff import validate_handoff
from lab.core.models import classify_runtime_failure
from lab.core.memory import write_canonical
def test_handoff_rejects_agent_decision():
 assert 'agent_cannot_set_supervisor_decision' in validate_handoff({'campaign_id':'c','episode_id':'e','status':'candidate_ready','candidate_commit':'x','supervisor_decision':'ACCEPT'})
def test_runtime_classification():
 assert classify_runtime_failure(TimeoutError('timed out')) == 'AGENT_TIMEOUT'
 assert classify_runtime_failure(ValueError('bad')) == 'AGENT_BACKEND_ERROR'
def test_canonical_requires_supervisor():
 with tempfile.TemporaryDirectory() as d:
  try: write_canonical(Path(d), {'campaign_id':'c'})
  except ValueError: pass
  else: assert False
