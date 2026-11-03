"""Paired parent/edited relational contract engineering checks (G3a).

The synthetic network is a deterministic pseudo-random float32 MLP, not a learned
controller; it exercises the import and relational query machinery only.
"""
import copy
from fractions import Fraction
import importlib.util
import json
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
from skillforge.wspl_import import import_bundle
from skillforge.wspl_relational import check_relation, check_relational_suite, preservation_contract

EXAMPLES = ROOT / 'examples/wspl'
WIDTHS = [3, 6, 6, 2]


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def synthetic_bundle(seed=20261103):
    rng = random.Random(seed)
    port = lambda name: {'name': name, 'type': {'kind': 'Real', 'unit': '1', 'lower': '-1', 'upper': '1'}}
    layers = []
    for i, (a, b) in enumerate(zip(WIDTHS, WIDTHS[1:])):
        layers.append({'activation': 'linear' if i == len(WIDTHS) - 2 else 'relu',
                       'weight': [[f32(rng.uniform(-1, 1)).hex() for _ in range(a)] for _ in range(b)],
                       'bias': [f32(rng.uniform(-0.5, 0.5)).hex() for _ in range(b)]})
    return {'format': 'wspl.mlp_import.v0', 'dtype': 'float32', 'tensor_layout': 'out_by_in',
            'inputs': [port(f'x{i}') for i in range(WIDTHS[0])], 'outputs': [port(f'y{i}') for i in range(WIDTHS[-1])],
            'layers': layers, 'preprocessing': [], 'postprocessing': [],
            'provenance': {'kind': 'synthetic', 'seed': seed,
                           'note': 'Deterministic pseudo-random float32 weights; not trained, not a learned baseline.'}}


def edit(bundle, layer, field, index, delta):
    edited = copy.deepcopy(bundle)
    values = edited['layers'][layer][field]
    if field == 'bias':
        values[index] = f32(float.fromhex(values[index]) + delta).hex()
    else:
        row, column = index
        values[row][column] = f32(float.fromhex(values[row][column]) + delta).hex()
    edited['provenance'] = {**edited['provenance'], 'edit': [layer, field, str(index), delta]}
    return edited


def side(terms=None, bias='0'):
    return {'terms': terms or {}, 'bias': str(bias), 'unit': '1'}


@unittest.skipUnless(importlib.util.find_spec('z3'), 'optional Z3 unavailable')
class RelationalTests(unittest.TestCase):
    def setUp(self):
        self.bundle = synthetic_bundle()
        self.parent = import_bundle(self.bundle)

    def assert_replayed(self, r):
        self.assertEqual(r['status'], 'REFUTED', r)
        self.assertTrue(r['witness_validated'])

    def test_identical_and_reimported_networks_are_preserved(self):
        again = import_bundle(synthetic_bundle())
        r = check_relation(self.parent, again, preservation_contract(self.parent, again))
        self.assertEqual(r['status'], 'PROVED', r)
        self.assertTrue(r['nonvacuous'])

    def test_compiled_and_imported_examples_are_preserved(self):
        for stem in ('brake_patch', 'approach_p'):
            with self.subTest(stem=stem):
                compiled = ReLUArtifact.load(EXAMPLES / f'{stem}.relu.json')
                imported = import_bundle(json.loads((EXAMPLES / f'{stem}.compiled_as_import.json').read_text()))
                r = check_relation(compiled, imported, preservation_contract(compiled, imported))
                self.assertEqual(r['status'], 'PROVED', r)

    def test_hidden_edit_is_refuted_globally_and_preserved_where_the_unit_is_inactive(self):
        weights = [[Fraction(float.fromhex(v)) for v in row] for row in self.bundle['layers'][0]['weight']]
        biases = [Fraction(float.fromhex(v)) for v in self.bundle['layers'][0]['bias']]
        unit = next(k for k in range(len(biases)) if sum(abs(w) for w in weights[k]) > biases[k] + Fraction(1, 2))
        edited = import_bundle(edit(self.bundle, 0, 'bias', unit, 0.25))
        new_bias = Fraction(float.fromhex(edit(self.bundle, 0, 'bias', unit, 0.25)['layers'][0]['bias'][unit]))
        self.assertNotEqual(new_bias, biases[unit])
        r = check_relation(self.parent, edited, preservation_contract(self.parent, edited))
        self.assert_replayed(r)
        x = r['counterexample']['inputs']
        self.assertNotEqual(self.parent.evaluate(x), edited.evaluate(x))
        # Region where the edited unit's pre-activation is <= 0 for both biases.
        region = {'op': 'le', 'left': side({f'input.x{i}': str(w) for i, w in enumerate(weights[unit])}),
                  'right': side(bias=-max(biases[unit], new_bias))}
        r = check_relation(self.parent, edited, preservation_contract(self.parent, edited, region=region))
        self.assertEqual(r['status'], 'PROVED', r)
        self.assertTrue(r['nonvacuous'])

    def test_output_bias_shift_and_tolerances(self):
        shift = 2 ** -10
        edited = import_bundle(edit(self.bundle, 2, 'bias', 0, shift))
        contract = lambda **kw: preservation_contract(self.parent, edited, **kw)
        self.assert_replayed(check_relation(self.parent, edited, contract()))
        self.assertEqual(check_relation(self.parent, edited, contract(tolerance=str(Fraction(shift))))['status'], 'PROVED')
        self.assert_replayed(check_relation(self.parent, edited, contract(tolerance=str(Fraction(shift) / 2))))
        self.assertEqual(check_relation(self.parent, edited, contract(outputs=['y1']))['status'], 'PROVED')

    def test_relative_repair_bounds(self):
        edited = import_bundle(edit(self.bundle, 2, 'bias', 0, -(2 ** -10)))
        base = preservation_contract(self.parent, edited)

        def relative(margin):
            return {**base, 'name': 'lowered', 'ensures': {
                'op': 'le', 'left': side({'edited.y0': '1'}), 'right': side({'parent.y0': '1'}, -margin)}}
        self.assertEqual(check_relation(self.parent, edited, relative(Fraction(1, 1024)))['status'], 'PROVED')
        self.assert_replayed(check_relation(self.parent, edited, relative(Fraction(1, 512))))

    def test_invalid_and_unknown_cases(self):
        edited = import_bundle(edit(self.bundle, 2, 'bias', 0, 2 ** -10))
        good = preservation_contract(self.parent, edited)
        stale = {**good, 'edited_sha256': '0' * 64}
        self.assertEqual(check_relation(self.parent, edited, stale)['status'], 'INVALID')
        vacuous = preservation_contract(self.parent, edited, region={'op': 'ge', 'left': side({'input.x0': '1'}), 'right': side(bias=2)})
        r = check_relation(self.parent, edited, vacuous)
        self.assertEqual(r['status'], 'INVALID')
        self.assertFalse(r['nonvacuous'])
        unknown_port = {**good, 'ensures': {'op': 'eq', 'left': side({'edited.zz': '1'}), 'right': side()}}
        self.assertEqual(check_relation(self.parent, edited, unknown_port)['status'], 'INVALID')
        wider = copy.deepcopy(self.bundle)
        wider['inputs'][0]['type']['upper'] = '2'
        other = import_bundle(wider)
        r = check_relation(self.parent, other, preservation_contract(self.parent, other))
        self.assertEqual(r['status'], 'INVALID')
        self.assertIn('input ports', r['reason'])
        self.assertEqual(check_relation(self.parent, edited, good, timeout_ms=0)['status'], 'UNKNOWN')
        self.assertEqual(check_relation(self.parent, edited, good, max_parameters=10)['status'], 'UNKNOWN')
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            preservation_contract(self.parent, edited, tolerance='-1')

    def test_suite_and_cli(self):
        edited = import_bundle(edit(self.bundle, 2, 'bias', 0, 2 ** -10))
        exact = preservation_contract(self.parent, edited, name='exact')
        banded = preservation_contract(self.parent, edited, name='banded', tolerance=str(Fraction(2 ** -10)))
        suite = {'format': 'wspl.relational_suite.v0', 'parent_sha256': self.parent.sha256, 'edited_sha256': edited.sha256,
                 'contracts': [{k: c[k] for k in ('name', 'assumptions', 'ensures')} for c in (banded, exact)]}
        report = check_relational_suite(self.parent, edited, suite)
        self.assertEqual(report['status'], 'REFUTED')
        self.assertEqual([r['status'] for r in report['results']], ['PROVED', 'REFUTED'])
        env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.parent.save(tmp / 'parent.json')
            edited.save(tmp / 'edited.json')
            (tmp / 'suite.json').write_text(json.dumps(suite))
            run = lambda out, s: subprocess.run([sys.executable, '-m', 'skillforge.wspl_relational', str(tmp / 'parent.json'),
                                                 str(tmp / 'edited.json'), str(s), '--output', str(tmp / out)],
                                                capture_output=True, text=True, env=env)
            self.assertEqual(run('r1.json', tmp / 'suite.json').returncode, 2)
            suite['contracts'] = suite['contracts'][:1]
            (tmp / 'banded.json').write_text(json.dumps(suite))
            self.assertEqual(run('r2.json', tmp / 'banded.json').returncode, 0)
            self.assertEqual(run('r2.json', tmp / 'banded.json').returncode, 1)


class ModuleBoundaryTests(unittest.TestCase):
    def test_relational_checker_does_not_import_the_compiler_or_importer(self):
        text = (ROOT / 'src/skillforge/wspl_relational.py').read_text(encoding='utf-8')
        self.assertNotIn('wspl_compile', text)
        self.assertNotIn('wspl_import', text)


if __name__ == '__main__':
    unittest.main()
