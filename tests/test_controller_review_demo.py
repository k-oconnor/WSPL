"""Check review orchestration, witness persistence and fail-closed outcomes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from demo_controller_review import run_demo


@unittest.skipUnless(importlib.util.find_spec('z3'), 'optional Z3 unavailable')
class ControllerReviewDemoTests(unittest.TestCase):
    def test_review_persists_replayed_violation_and_integrity_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'review'
            report = run_demo(output)
            self.assertEqual(report['status'], 'EXPECTED_FIXTURE_OUTCOMES')
            self.assertEqual([s['status'] for s in report['stages']], ['PROVED', 'REFUTED', 'PROVED'])
            broken = json.loads((output / 'reduced_braking_gain.verification.json').read_text())
            failures = [r for r in broken['results'] if r['status'] == 'REFUTED']
            self.assertTrue(failures)
            self.assertTrue(all(r['witness_validated'] for r in failures))
            self.assertEqual(report['stages'][0]['artifact_sha256'], report['stages'][2]['artifact_sha256'])
            inventory = json.loads((output / 'manifest.json').read_text())['files']
            self.assertEqual(set(inventory), {p.name for p in output.iterdir()} - {'manifest.json'})
            for name, digest in inventory.items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                run_demo(output)

    def test_no_solver_budget_never_reports_demo_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_demo(Path(tmp) / 'review', timeout_ms=0)
            self.assertEqual(report['status'], 'INCOMPLETE_OR_UNEXPECTED')
            self.assertTrue(all(s['status'] == 'UNKNOWN' for s in report['stages']))

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'), 'optional Torch/NumPy unavailable')
    def test_empirical_path_keeps_effects_separate_and_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_demo(Path(tmp) / 'review', include_opaque=True)
            self.assertEqual(report['status'], 'EXPECTED_FIXTURE_OUTCOMES')
            self.assertEqual(report['opaque']['outcomes'], ['OBSERVED_EFFECT', 'NO_OBSERVED_EFFECT'])
            self.assertTrue(report['opaque']['exact_replay'])
            self.assertFalse(report['scientific_run'])


if __name__ == '__main__':
    unittest.main()
