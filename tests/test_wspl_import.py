"""Frozen-tensor import engineering checks (G3a); fixtures are not learned baselines."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from skillforge.wspl_artifact import ReLUArtifact, canonical_bytes
from skillforge.wspl_import import bundle_from_artifact, import_bundle

EXAMPLES = ROOT / 'examples/wspl'


def committed(stem):
    return ReLUArtifact.load(EXAMPLES / f'{stem}.relu.json')


def fixture(stem):
    return json.loads((EXAMPLES / f'{stem}.compiled_as_import.json').read_text(encoding='utf-8'))


class ImportFidelityTests(unittest.TestCase):
    def test_fixture_bundles_import_to_the_committed_layers(self):
        for stem in ('brake_patch', 'approach_p'):
            with self.subTest(stem=stem):
                artifact = import_bundle(fixture(stem))
                self.assertEqual(artifact.to_dict()['layers'], committed(stem).to_dict()['layers'])
                self.assertEqual(artifact.input_names, committed(stem).input_names)
                self.assertEqual(artifact.output_names, committed(stem).output_names)

    def test_fixture_bundles_are_reproducible_and_labelled(self):
        for stem in ('brake_patch', 'approach_p'):
            with self.subTest(stem=stem):
                bundle = fixture(stem)
                self.assertIn('Not a learned network', bundle['provenance']['note'])
                self.assertEqual(bundle['provenance']['source_artifact_sha256'], committed(stem).sha256)
                self.assertEqual(bundle_from_artifact(committed(stem), provenance=bundle['provenance']), bundle)

    def test_layouts_give_identical_layers(self):
        provenance = fixture('approach_p')['provenance']
        transposed = bundle_from_artifact(committed('approach_p'), provenance=provenance, tensor_layout='in_by_out')
        self.assertNotEqual(transposed['layers'][0]['weight'], fixture('approach_p')['layers'][0]['weight'])
        a, b = import_bundle(fixture('approach_p')), import_bundle(transposed)
        self.assertEqual(a.to_dict()['layers'], b.to_dict()['layers'])
        self.assertIn('transposed', b.to_dict()['source_map']['import']['stored_layers'])

    def test_identity_and_metadata(self):
        bundle = fixture('brake_patch')
        d = import_bundle(bundle).to_dict()
        self.assertEqual(d['compiler'], 'wspl.mlp_import.v0')
        self.assertEqual(d['allocation'], {'mode': 'imported', 'widths': [3, 4, 2, 1], 'parameters': 29})
        self.assertEqual(d['source_sha256'], hashlib.sha256(canonical_bytes(bundle)).hexdigest())
        record = d['source_map']['import']
        self.assertEqual(record['dtype'], 'float32')
        self.assertEqual(record['provenance'], bundle['provenance'])
        self.assertEqual(record['tensor_sha256'][0]['weight'],
                         hashlib.sha256(canonical_bytes(bundle['layers'][0]['weight'])).hexdigest())

    def test_imported_artifacts_round_trip_through_save_and_load(self):
        artifact = import_bundle(fixture('brake_patch'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.json'
            artifact.save(path)
            self.assertEqual(ReLUArtifact.load(path, expected_sha256=artifact.sha256).to_dict(), artifact.to_dict())

    def test_loader_still_rejects_other_allocation_modes(self):
        d = import_bundle(fixture('brake_patch')).to_dict()
        d['allocation']['mode'] = 'fixed'
        with self.assertRaisesRegex(ValueError, 'allocation metadata'):
            ReLUArtifact(d)


class SourceBindingTests(unittest.TestCase):
    def test_imported_artifacts_are_not_bound_to_scalar_ir(self):
        from skillforge.wspl_parse import parse_module
        imported = import_bundle(fixture('brake_patch'))
        parsed = parse_module((EXAMPLES / 'brake_patch.wspl').read_text(encoding='utf-8'))
        with self.assertRaisesRegex(ValueError, 'not compiled from this module'):
            parsed.contract_suite(imported)
        from skillforge.wspl_equiv import check_equivalence
        source = json.loads((EXAMPLES / 'brake_patch.core.json').read_text(encoding='utf-8'))
        self.assertEqual(check_equivalence(source, imported)['status'], 'INVALID')

    @unittest.skipUnless(importlib.util.find_spec('z3'), 'optional Z3 unavailable')
    def test_existing_contracts_prove_on_imported_weights(self):
        from skillforge.wspl_verify import verify_suite
        for stem, contracts in (('brake_patch', 'brake'), ('approach_p', 'approach')):
            with self.subTest(stem=stem):
                imported = import_bundle(fixture(stem))
                suite = json.loads((EXAMPLES / f'{contracts}.contracts.json').read_text(encoding='utf-8'))
                suite['artifact_sha256'] = imported.sha256
                self.assertEqual(verify_suite(imported, suite)['status'], 'PROVED')


class RejectionTests(unittest.TestCase):
    def mutated(self, change):
        bundle = copy.deepcopy(fixture('brake_patch'))
        change(bundle)
        return bundle

    def assert_rejected(self, change, message):
        with self.assertRaisesRegex(ValueError, message):
            import_bundle(self.mutated(change))

    def set_bias(self, value, dtype='float32'):
        def change(b):
            b['dtype'] = dtype
            b['layers'][0]['bias'][0] = value
        return change

    def test_number_encoding(self):
        for value in ('1.5', '0.25', 'nan', 'inf', '-infinity', '0x1.8', 1.5, '0X1P+0 '):
            with self.subTest(value=value):
                self.assert_rejected(self.set_bias(value), 'hex float string required')
        self.assert_rejected(self.set_bias('0x1.999999999999ap-4'), 'not exactly representable in float32')
        self.assertIsNotNone(import_bundle(self.mutated(self.set_bias('0x1.999999999999ap-4', 'float64'))))
        self.assert_rejected(self.set_bias('0x1p+200'), 'overflows float32')
        self.assert_rejected(self.set_bias('0x1p+1024', 'float64'), 'overflows float64')

    def test_structure_and_profiles(self):
        self.assert_rejected(lambda b: b.update(dtype='bfloat16'), 'dtype')
        self.assert_rejected(lambda b: b.update(tensor_layout='auto'), 'never guessed')
        self.assert_rejected(lambda b: b.update(tensor_layout='in_by_out'), 'check tensor_layout')
        self.assert_rejected(lambda b: b['layers'][0].update(activation='tanh'), 'only hidden relu')
        self.assert_rejected(lambda b: b['layers'][-1].update(activation='relu'), 'only hidden relu')
        self.assert_rejected(lambda b: b['layers'][0]['weight'][0].pop(), 'ragged')
        self.assert_rejected(lambda b: b['layers'][0]['bias'].pop(), 'bias length')
        self.assert_rejected(lambda b: b.update(preprocessing=[{'input': 'd', 'scale': '0x1p+1'}]), 'G3b')
        self.assert_rejected(lambda b: b.update(postprocessing=[{}]), 'G3b')
        self.assert_rejected(lambda b: b.update(provenance={}), 'provenance')
        self.assert_rejected(lambda b: b.update(extra=1), 'bundle with fields')
        self.assert_rejected(lambda b: b['inputs'][0]['type'].update(upper=2), 'exact rational strings')
        self.assert_rejected(lambda b: b['inputs'][1].update(name='d'), 'duplicate')


class CommandLineTests(unittest.TestCase):
    def test_cli_imports_exclusively(self):
        env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'imported.json'
            args = [sys.executable, '-m', 'skillforge.wspl_import', str(EXAMPLES / 'brake_patch.compiled_as_import.json'),
                    '--output', str(out)]
            first = subprocess.run(args, capture_output=True, text=True, env=env)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(json.loads(first.stdout)['allocation']['mode'], 'imported')
            self.assertEqual(ReLUArtifact.load(out).to_dict()['layers'], committed('brake_patch').to_dict()['layers'])
            self.assertEqual(subprocess.run(args, capture_output=True, text=True, env=env).returncode, 1)


if __name__ == '__main__':
    unittest.main()
