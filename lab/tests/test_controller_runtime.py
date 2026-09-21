import shutil
import tempfile
import unittest
from pathlib import Path

from lab.core.controller import LabController


class _Runner:
    def __init__(self): self.paused=False;self.stopped=False
    def pause(self): self.paused=True
    def stop(self): self.stopped=True


class ControllerRuntimeTests(unittest.TestCase):
    def test_pause_stop_and_harvest_are_persistent(self):
        root=Path(tempfile.mkdtemp(prefix="aka-lab-controller-"))
        try:
            (root/"lab"/"campaigns"/"fixture").mkdir(parents=True)
            ctrl=LabController(root);ctrl.configure_local("rms_norm_train","cuda_cpp","fixture");ctrl.runner=_Runner();ctrl.run_config={"candidate_root":"fixture-candidate"}
            self.assertTrue(ctrl.pause_optimization()["pause_requested"]);self.assertTrue(ctrl.runner.paused)
            self.assertTrue(ctrl.stop_optimization()["stop_requested"]);self.assertTrue(ctrl.runner.stopped)
            checkpoint=Path(ctrl.harvest("fixture"))
            self.assertTrue((checkpoint/"checkpoint.json").exists())
            self.assertTrue((checkpoint/"CHECKPOINT.md").exists())
        finally: shutil.rmtree(root)
