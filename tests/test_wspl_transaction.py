"""Transactional patch commit/rollback engineering checks (G3b); no learned parents."""
import copy
import json
import importlib.util
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from skillforge.wspl_artifact import ReLUArtifact
from skillforge.wspl_compile import compile_scalar
from skillforge.wspl_import import import_bundle
from skillforge.wspl_parse import parse_module
from skillforge.wspl_transaction import decide, patch_transaction, replay_transaction

EXAMPLES = ROOT / 'examples/wspl'
BINDINGS = {'d': 'input.d', 'v': 'input.v', 'a_base': 'parent.a'}
REPLACE = {'a': 'a'}


def cruise():
    return compile_scalar(parse_module((EXAMPLES / 'cruise_baseline.wspl').read_text(encoding='utf-8')).program)


def brake():
    return ReLUArtifact.load(EXAMPLES / 'brake_patch.relu.json')


def side(terms=None, bias='0'):
    return {'terms': terms or {}, 'bias': bias, 'unit': '1'}


def brake_io():
    suite = json.loads((EXAMPLES / 'brake.contracts.json').read_text(encoding='utf-8'))
    return [c for c in suite['contracts'] if c['name'] in ('actuator_bound', 'emergency_brake')]


CLEAR_PATH = {'name': 'clear_path_preserved',
              'assumptions': {'op': 'ge', 'left': side({'input.d': '1', 'input.v': '-1/2'}), 'right': side(bias='2/5')},
              'ensures': {'op': 'eq', 'left': side({'edited.a': '1'}), 'right': side({'parent.a': '1'})}}
EVERYWHERE = {'name': 'unchanged_everywhere', 'assumptions': {'op': 'true'},
              'ensures': {'op': 'eq', 'left': side({'edited.a': '1'}), 'right': side({'parent.a': '1'})}}


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def unbounded_parent(seed=11):
    """Untrained pseudo-random float32 parent over (d, v) whose output can leave [-1, 1]."""
    rng = random.Random(seed)
    ports = [{'name': 'd', 'type': {'kind': 'Real', 'unit': '1', 'lower': '0', 'upper': '2'}},
             {'name': 'v', 'type': {'kind': 'Real', 'unit': '1', 'lower': '0', 'upper': '1'}}]
    layer = lambda a, b, act: {'activation': act, 'weight': [[f32(rng.uniform(-3, 3)).hex() for _ in range(a)] for _ in range(b)],
                               'bias': [f32(rng.uniform(-1, 1)).hex() for _ in range(b)]}
    return import_bundle({'format': 'wspl.mlp_import.v0', 'dtype': 'float32', 'tensor_layout': 'out_by_in',
                          'inputs': ports, 'outputs': [{'name': 'a', 'type': {'kind': 'Real', 'unit': '1', 'lower': '-1', 'upper': '1'}}],
                          'layers': [layer(2, 4, 'relu'), layer(4, 1, 'linear')], 'preprocessing': [], 'postprocessing': [],
                          'provenance': {'kind': 'synthetic', 'seed': seed, 'note': 'untrained; output bound deliberately unestablished'}})


class DecisionRuleTests(unittest.TestCase):
    def test_three_outcomes(self):
        self.assertEqual(decide(['PROVED', 'PROVED']), 'COMMITTED')
        self.assertEqual(decide([]), 'COMMITTED')
        self.assertEqual(decide(['PROVED', 'UNKNOWN']), 'UNRESOLVED')
        self.assertEqual(decide(['UNKNOWN', 'REFUTED']), 'REJECTED')
        self.assertEqual(decide(['PROVED', 'INVALID']), 'REJECTED')


@unittest.skipUnless(importlib.util.find_spec('z3'), 'optional Z3 unavailable')
class TransactionTests(unittest.TestCase):
    def run_transaction(self, parent=None, **kw):
        kw.setdefault('io_contracts', brake_io())
        kw.setdefault('relational_contracts', [CLEAR_PATH])
        return patch_transaction(parent or cruise(), brake(), bindings=BINDINGS, replace=REPLACE, **kw)

    def test_commit_when_obligations_and_contracts_prove(self):
        result = self.run_transaction()
        m = result.manifest
        self.assertEqual(result.decision, 'COMMITTED', [(r['name'], r['status'], r.get('reason')) for r in m['results']])
        self.assertEqual(result.artifact.sha256, result.candidate.sha256)
        self.assertEqual(m['returned_sha256'], m['candidate_sha256'])
        self.assertEqual(m['contract_counts'], {'interface': 1, 'io': 2, 'relational': 1})
        self.assertEqual([r['kind'] for r in m['results']], ['interface', 'io', 'io', 'relational'])
        self.assertEqual(m['label'], 'output-patch augmentation')
        self.assertGreater(m['placement_rows_checked'], 0)
        self.assertTrue(m['parent_unchanged'])
        self.assertEqual(replay_transaction(m, cruise(), brake())['consistent'], True)
        self.assertEqual(replay_transaction(m, cruise(), brake(), rerun=True)['decision'], 'COMMITTED')

    def test_unestablished_parent_bound_rejects_and_returns_parent(self):
        parent = unbounded_parent()
        before = parent.sha256
        result = self.run_transaction(parent=parent)
        interface = [r for r in result.manifest['results'] if r['kind'] == 'interface']
        self.assertEqual(interface[0]['status'], 'REFUTED', interface)
        self.assertTrue(interface[0]['witness_validated'])
        self.assertEqual(result.decision, 'REJECTED')
        self.assertEqual(result.artifact.sha256, before)
        self.assertEqual(result.manifest['returned_sha256'], before)
        self.assertIsNotNone(result.manifest['candidate_sha256'])
        self.assertNotEqual(result.manifest['candidate_sha256'], before)

    def test_false_preservation_claim_rejects_with_counterexample(self):
        result = self.run_transaction(relational_contracts=[CLEAR_PATH, EVERYWHERE])
        self.assertEqual(result.decision, 'REJECTED')
        failed = [r for r in result.manifest['results'] if r['status'] != 'PROVED']
        self.assertEqual([r['name'] for r in failed], ['unchanged_everywhere'])
        self.assertIn('counterexample', failed[0])
        self.assertEqual(result.artifact.sha256, cruise().sha256)

    def test_budget_exhaustion_is_unresolved_not_rejected(self):
        result = self.run_transaction(timeout_ms=0)
        self.assertEqual({r['status'] for r in result.manifest['results']}, {'UNKNOWN'})
        self.assertEqual(result.decision, 'UNRESOLVED')
        self.assertEqual(result.artifact.sha256, cruise().sha256)

    def test_invalid_contract_rejects(self):
        bad = {**EVERYWHERE, 'name': 'bad_port', 'ensures': {'op': 'eq', 'left': side({'edited.zz': '1'}), 'right': side()}}
        result = self.run_transaction(relational_contracts=[bad])
        self.assertEqual(result.decision, 'REJECTED')
        self.assertEqual(result.manifest['results'][-1]['status'], 'INVALID')

    def test_zero_behavioural_contracts_is_visible(self):
        result = self.run_transaction(io_contracts=[], relational_contracts=[])
        self.assertEqual(result.decision, 'COMMITTED')
        self.assertEqual(result.manifest['contract_counts'], {'interface': 1, 'io': 0, 'relational': 0})

    def test_invalid_composition_rejects_without_candidate(self):
        result = patch_transaction(cruise(), brake(), bindings={**BINDINGS, 'v': 'input.d'}, replace=REPLACE)
        self.assertEqual(result.decision, 'REJECTED')
        self.assertIsNone(result.candidate)
        self.assertIsNone(result.manifest['candidate_sha256'])
        self.assertIn('composition invalid', result.manifest['reason'])
        self.assertEqual(result.artifact.sha256, cruise().sha256)

    def test_replay_detects_stale_or_altered_records(self):
        m = self.run_transaction().manifest
        with self.assertRaisesRegex(ValueError, 'differs from the manifest'):
            replay_transaction(m, unbounded_parent(), brake())
        for key, value in (('bindings', {**BINDINGS, 'd': 'input.v'}), ('decision', 'REJECTED'), ('candidate_sha256', '0' * 64)):
            with self.subTest(field=key):
                altered = copy.deepcopy(m)
                altered[key] = value
                with self.assertRaisesRegex(ValueError, 'altered'):
                    replay_transaction(altered, cruise(), brake())
        altered = copy.deepcopy(m)
        altered['results'][0]['status'] = 'REFUTED'
        with self.assertRaisesRegex(ValueError, 'altered'):
            replay_transaction(altered, cruise(), brake())

    def test_cli_writes_artifact_only_on_commit(self):
        env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cruise().save(tmp / 'parent.json')
            unbounded_parent().save(tmp / 'unbounded.json')
            spec = {'format': 'wspl.patch_spec.v0', 'bindings': BINDINGS, 'replace': REPLACE,
                    'io_contracts': brake_io(), 'relational_contracts': [CLEAR_PATH]}
            (tmp / 'spec.json').write_text(json.dumps(spec))
            run = lambda parent, tag: subprocess.run(
                [sys.executable, '-m', 'skillforge.wspl_transaction', str(tmp / parent), str(EXAMPLES / 'brake_patch.relu.json'),
                 str(tmp / 'spec.json'), '--manifest-output', str(tmp / f'{tag}.manifest.json'),
                 '--artifact-output', str(tmp / f'{tag}.edited.json')], capture_output=True, text=True, env=env)
            ok = run('parent.json', 'ok')
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertTrue((tmp / 'ok.edited.json').exists())
            self.assertEqual(json.loads((tmp / 'ok.manifest.json').read_text())['decision'], 'COMMITTED')
            rejected = run('unbounded.json', 'bad')
            self.assertEqual(rejected.returncode, 2, rejected.stderr)
            self.assertFalse((tmp / 'bad.edited.json').exists())
            self.assertEqual(run('parent.json', 'ok').returncode, 1)


if __name__ == '__main__':
    unittest.main()
