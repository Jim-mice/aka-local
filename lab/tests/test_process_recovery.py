import os, shutil, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from lab.runtime.process_registry import ProcessRegistry, ManagedProcessLauncher
from lab.runtime.run_state import RunStore
from lab.core.controller import LabController

class ProcessRecoveryTests(unittest.TestCase):
    def test_process_tree_isolation_and_emergency_stop(self):
        temp=Path(tempfile.mkdtemp(prefix="aka-proc-"))
        try:
            registry=ProcessRegistry(temp,"test-run"); code="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',\"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)\"]); time.sleep(60)"
            flags=getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0); parent=subprocess.Popen([sys.executable,"-c",code],creationflags=flags);registry.attach(parent,kind="fixture_parent")
            unrelated=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],creationflags=flags)
            time.sleep(.5);registry.terminate_tree();time.sleep(.5)
            self.assertIsNotNone(parent.poll());self.assertIsNone(unrelated.poll())
            subprocess.run(["taskkill","/PID",str(unrelated.pid),"/T","/F"],capture_output=True);unrelated.wait(timeout=5)
        finally:shutil.rmtree(temp)
    def test_resume_manifest_roundtrip_and_stale_detection(self):
        temp=Path(tempfile.mkdtemp(prefix="aka-run-"))
        try:
            store=RunStore(temp);store.write_run({"campaign_id":"fixture","candidate_root":"C:/candidate","incumbent":"C:/inc","operator_id":"rms_norm_train","backend_id":"cuda_cpp"});store.save(state="PAUSED",last_safe_point="AFTER_CORRECTNESS",current_experiment_index=2,resume_safe=True)
            recovered=RunStore.recoverable(temp);self.assertEqual(recovered[0]["state"],"PAUSED");self.assertEqual(recovered[0]["last_safe_point"],"AFTER_CORRECTNESS")
            registry=ProcessRegistry(store.dir,store.run_id);registry.data["processes"].append({"pid":999999,"state":"ACTIVE"});registry._save();self.assertEqual(registry.mark_stale_after_restart()[0]["state"],"STALE_PROCESS_RECORD")
        finally:shutil.rmtree(temp)
    def test_real_cross_process_filesystem_only_recovery(self):
        temp=Path(tempfile.mkdtemp(prefix="aka-cross-process-"))
        try:
            code="from lab.runtime.run_state import RunStore; import sys; s=RunStore(sys.argv[1]); s.write_run({'campaign_id':'fixture','candidate_root':'C:/candidate','incumbent':'C:/inc'}); s.save(state='PAUSED',last_safe_point='AFTER_DEV_BENCHMARK',resume_safe=True)"
            subprocess.run([sys.executable,"-c",code,str(temp)],check=True)
            code2="from lab.runtime.run_state import RunStore; import sys; x=RunStore.recoverable(sys.argv[1]); assert len(x)==1 and x[0]['state']=='PAUSED' and x[0]['last_safe_point']=='AFTER_DEV_BENCHMARK'; print('REAL_CROSS_PROCESS_RECOVERY_PASS')"
            result=subprocess.run([sys.executable,"-c",code2,str(temp)],capture_output=True,text=True,check=True)
            self.assertIn("REAL_CROSS_PROCESS_RECOVERY_PASS",result.stdout)
        finally:shutil.rmtree(temp)
    def test_environment_mismatch_blocks_resume(self):
        temp=Path(tempfile.mkdtemp(prefix="aka-env-"))
        try:
            store=RunStore(temp/"lab");store.write_run({"campaign_id":"fixture","candidate_root":"C:/missing","incumbent":"C:/missing","operator_id":"rms_norm_train","backend_id":"cuda_cpp"});store.save(state="PAUSED",environment_fingerprint="deliberately-wrong",resume_safe=True)
            ctrl=LabController(temp)
            with self.assertRaisesRegex(RuntimeError,"ENVIRONMENT_CHANGED"):ctrl.resume_recovered(store.run_id)
        finally:shutil.rmtree(temp)
