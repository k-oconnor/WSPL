"""Exercise the actual replacement runner with synthetic model and simulator."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_fresh_robot as fixture
import run_fresh_robot_parent_v2 as replacement


class CountingSimulator(fixture.FakeSimulator):
    created = 0
    finished = 0

    def __init__(self, *args):
        super().__init__(*args)
        CountingSimulator.created += 1
        self.starts = 0

    def call(self, r):
        if r['op'] == 'start':
            self.starts += 1
            if self.starts != 1:
                raise AssertionError('Scene reused in same worker')
        return super().call(r)

    def close(self):
        CountingSimulator.finished += 1
        super().close()


class FreshRobotV2Tests(unittest.TestCase):
    def run_fake(self, out):
        with patch.object(fixture, 'runner', replacement), patch.object(fixture, 'FakeSimulator', CountingSimulator):
            fixture.FreshRobotTests().run_fake(out)

    def test_independent_process_per_training_and_reserved_mission(self):
        CountingSimulator.created = CountingSimulator.finished = 0
        CountingSimulator.failure = False
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self.run_fake(out)
            self.assertEqual(CountingSimulator.created, 6)  # 2 teacher + 2x2 evaluated.
            self.assertEqual(CountingSimulator.finished, 6)
            self.assertEqual(len(list(out.glob('*.rpc'))), 6)
            self.assertTrue(json.loads((out/'results.json').read_text())['summary']['parent_ready'])

    def test_failure_closes_worker_and_does_not_retry(self):
        CountingSimulator.created = CountingSimulator.finished = 0
        CountingSimulator.failure = True
        try:
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                with self.assertRaisesRegex(ValueError, 'teacher failed'):
                    self.run_fake(out)
                self.assertEqual(CountingSimulator.created, 1)
                self.assertEqual(CountingSimulator.finished, 1)
                self.assertFalse((out/'training.jsonl').exists())
        finally:
            CountingSimulator.failure = False


if __name__ == '__main__':
    unittest.main()
