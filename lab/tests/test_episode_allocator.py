"""Episode namespace allocation is CPU-only filesystem inspection."""
import tempfile
import unittest
from pathlib import Path

from lab.gui import existing_episode_directories, next_episode_id


class EpisodeAllocatorTests(unittest.TestCase):
    def test_smoke_directories_occupy_the_same_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign=Path(temporary)/"campaign";episodes=campaign/"episodes"
            for name in ("e0001","e0002_luna_smoke","e0003_luna_smoke_full_evaluator"):
                (episodes/name).mkdir(parents=True)
            self.assertEqual(existing_episode_directories(campaign),[
                (1,"e0001"),(2,"e0002_luna_smoke"),(3,"e0003_luna_smoke_full_evaluator")])
            self.assertEqual(next_episode_id(campaign),"e0004")
            (episodes/"e0004_test").mkdir()
            self.assertEqual(next_episode_id(campaign),"e0005")

