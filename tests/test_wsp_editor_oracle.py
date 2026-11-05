"""Archived WSPL programs must reproduce the fixture recorded on unmodified main (a9d7510).

Fast suites always run. The 40 Go replays (1.48M parameters) run with
WSPL_ORACLE_FULL=1, or directly: python tools/wsp_editor_oracle.py --check FIXTURE.
"""
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import wsp_editor_oracle as oracle

FIXTURE = ROOT/'tests/fixtures/wsp_editor_oracle_v1.json'


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(FIXTURE.read_text(encoding='utf-8'))['cases']

    def check(self, suite):
        results, _ = oracle.collect((suite,))
        self.assertTrue(results)
        for key, record in results.items():
            with self.subTest(case=key):
                self.assertEqual(record, self.expected[key])

    def test_breakout_reproduces_archived_candidates(self):
        self.check('breakout')
        archived = oracle.archived_breakout_ids()
        self.assertEqual({k: self.expected[k]['candidate_id'] for k in archived}, archived)

    def test_kinetix(self):
        self.check('kinetix')

    def test_diffusion(self):
        self.check('diffusion')

    def test_repair_reproduces_archived_candidates(self):
        self.check('repair')
        for key, record in self.expected.items():
            if key.startswith('repair/'):
                self.assertEqual(record['candidate_id'], record['archived_candidate_id'])

    @unittest.skipUnless(os.environ.get('WSPL_ORACLE_FULL') == '1', 'set WSPL_ORACLE_FULL=1 for the 40 Go replays')
    def test_go(self):
        self.check('go')


if __name__ == '__main__':
    unittest.main()
