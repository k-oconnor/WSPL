"""Output-patch augmentation engineering checks (G3b); fixtures are compiled or synthetic, not learned."""
from fractions import Fraction as Q
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from skillforge.wspl_artifact import ReLUArtifact, canonical_bytes
from skillforge.wspl_compile import compile_scalar
from skillforge.wspl_compose import compose_output, composition_record, interface_obligations, verify_placement
from skillforge.wspl_import import bundle_from_artifact, import_bundle
from skillforge.wspl_parse import parse_module

EXAMPLES = ROOT / 'examples/wspl'
BRAKE_BINDINGS = {'d': 'input.d', 'v': 'input.v', 'a_base': 'parent.a'}
COMBINED = '''module CruiseWithBrake {
  input d: Real[0,2] normalized_distance;
  input v: Real[0,1] normalized_speed;
  output a: Real[-1,1] normalized_acceleration;
  let a_base: Real[-1,1] = clip(0.5 - v + 0.25*d, -1, 1);
  let urgency: Truth = clip((0.4 + 0.5*v - d)/0.4, 0, 1);
  a = clip(a_base - 2*urgency, -1, 1);
}
'''


def cruise():
    return compile_scalar(parse_module((EXAMPLES / 'cruise_baseline.wspl').read_text(encoding='utf-8')).program)


def brake():
    return ReLUArtifact.load(EXAMPLES / 'brake_patch.relu.json')


def compiled(text):
    return compile_scalar(parse_module(text).program)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def synthetic_parent(seed=7, widths=(3, 5, 2)):
    """Deterministic pseudo-random untrained float32 MLP with outputs y0, y1."""
    rng = random.Random(seed)
    port = lambda name: {'name': name, 'type': {'kind': 'Real', 'unit': '1', 'lower': '-1', 'upper': '1'}}
    layers = [{'activation': 'linear' if i == len(widths) - 2 else 'relu',
               'weight': [[f32(rng.uniform(-1, 1)).hex() for _ in range(a)] for _ in range(b)],
               'bias': [f32(rng.uniform(-0.5, 0.5)).hex() for _ in range(b)]}
              for i, (a, b) in enumerate(zip(widths, widths[1:]))]
    return import_bundle({'format': 'wspl.mlp_import.v0', 'dtype': 'float32', 'tensor_layout': 'out_by_in',
                          'inputs': [port(f'x{i}') for i in range(widths[0])],
                          'outputs': [port(f'y{i}') for i in range(widths[-1])], 'layers': layers,
                          'preprocessing': [], 'postprocessing': [],
                          'provenance': {'kind': 'synthetic', 'seed': seed, 'note': 'untrained pseudo-random fixture'}})


SCALE_PATCH = '''module ScalePatch {
  input y_in: Real[-10,10] normalized_y;
  input x: Real[-1,1] normalized_x;
  output y: Real[-20,20] normalized_y;
  y = 0.5*y_in + relu(x);
}
'''
LINEAR_PATCH = SCALE_PATCH.replace('relu(x)', '0.25*x').replace('ScalePatch', 'LinearPatch')
LINEAR_PARENT = '''module LinearParent {
  input x0: Real[-1,1] normalized_x;
  input x1: Real[-1,1] normalized_x;
  input x2: Real[-1,1] normalized_x;
  output y0: Real[-1,1] normalized_y;
  output y1: Real[-1,1] normalized_y;
  y0 = 0.5*x0 - 0.25*x1;
  y1 = 0.5*x2;
}
'''


def grid(artifact, points=5):
    d = artifact.to_dict()
    axes = [[Q(p['type']['lower']) + (Q(p['type']['upper']) - Q(p['type']['lower'])) * Q(i, points - 1)
             for i in range(points)] for p in d['inputs']]
    for values in itertools.product(*axes):
        yield dict(zip(artifact.input_names, values))


def reference(parent, patch, bindings, replace, x):
    """Direct semantics: run the parent, feed the patch, substitute replaced outputs."""
    y = parent.evaluate(x)
    feed = {name: (x if source.startswith('input.') else y)[source.split('.', 1)[1]] for name, source in bindings.items()}
    patched = patch.evaluate(feed)
    return {**y, **{target: patched[name] for name, target in replace.items()}}


class CompositionSemanticsTests(unittest.TestCase):
    def check_case(self, parent, patch, bindings, replace):
        edited = compose_output(parent, patch, bindings=bindings, replace=replace)
        for x in grid(parent):
            self.assertEqual(edited.evaluate(x), reference(parent, patch, bindings, replace, x), x)
        self.assertGreater(verify_placement(edited, parent, patch), 0)
        return edited

    def test_brake_patch_on_compiled_and_imported_cruise(self):
        imported = import_bundle(bundle_from_artifact(cruise(), provenance={'note': 'compiled stand-in, not learned'}))
        for parent in (cruise(), imported):
            with self.subTest(parent=parent.to_dict()['compiler']):
                self.check_case(parent, brake(), BRAKE_BINDINGS, {'a': 'a'})

    def test_passthrough_hidden_and_linear_only_shapes(self):
        cases = [(synthetic_parent(), compiled(SCALE_PATCH)), (synthetic_parent(), compiled(LINEAR_PATCH)),
                 (compiled(LINEAR_PARENT), compiled(SCALE_PATCH)), (compiled(LINEAR_PARENT), compiled(LINEAR_PATCH))]
        self.assertEqual(len(compiled(LINEAR_PATCH).to_dict()['layers']), 1)
        self.assertEqual(len(compiled(LINEAR_PARENT).to_dict()['layers']), 1)
        for parent, patch in cases:
            with self.subTest(parent=parent.to_dict()['allocation']['widths'], patch=patch.to_dict()['allocation']['widths']):
                for bindings, replace in (({'y_in': 'parent.y0', 'x': 'input.x2'}, {'y': 'y0'}),
                                          ({'y_in': 'parent.y1', 'x': 'input.x0'}, {'y': 'y1'}),
                                          ({'y_in': 'parent.y1', 'x': 'input.x1'}, {'y': 'y0'})):
                    edited = self.check_case(parent, patch, bindings, replace)
                    self.assertEqual(edited.output_names, ('y0', 'y1'))

    def test_metadata_and_identity(self):
        edited = compose_output(cruise(), brake(), bindings=BRAKE_BINDINGS, replace={'a': 'a'})
        d = edited.to_dict()
        record = composition_record(cruise(), brake(), BRAKE_BINDINGS, {'a': 'a'})
        self.assertEqual(d['compiler'], 'wspl.compose_output.v0')
        self.assertEqual(d['allocation']['mode'], 'augmented')
        self.assertEqual(d['source_sha256'], hashlib.sha256(canonical_bytes(record)).hexdigest())
        composition = d['source_map']['composition']
        self.assertEqual(composition['label'], 'output-patch augmentation')
        self.assertEqual(composition['parent_sha256'], cruise().sha256)
        self.assertEqual(composition['added_depth'], len(d['allocation']['widths']) - 3)
        self.assertEqual(composition['added_parameters'], d['allocation']['parameters'] - cruise().to_dict()['allocation']['parameters'])
        self.assertEqual(composition['patch_output_types']['a'], brake().to_dict()['outputs'][0]['type'])
        self.assertEqual(d['inputs'], cruise().to_dict()['inputs'])
        self.assertEqual(d['outputs'], cruise().to_dict()['outputs'])
        with tempfile.TemporaryDirectory() as tmp:
            edited.save(Path(tmp) / 'e.json')
            self.assertEqual(ReLUArtifact.load(Path(tmp) / 'e.json').sha256, edited.sha256)
        again = compose_output(cruise(), brake(), bindings=dict(reversed(list(BRAKE_BINDINGS.items()))), replace={'a': 'a'})
        self.assertEqual(again.sha256, edited.sha256)

    def test_placement_check_detects_tampering(self):
        edited = compose_output(cruise(), brake(), bindings=BRAKE_BINDINGS, replace={'a': 'a'})
        for level in range(len(edited.to_dict()['layers'])):
            with self.subTest(level=level):
                d = edited.to_dict()
                d['layers'][level]['bias'][0] = str(Q(d['layers'][level]['bias'][0]) + Q(1, 1024))
                with self.assertRaisesRegex(ValueError, 'exact slice'):
                    verify_placement(ReLUArtifact(d), cruise(), brake())
        with self.assertRaisesRegex(ValueError, 'different parent'):
            verify_placement(edited, synthetic_parent(), brake())

    def test_interface_obligations(self):
        obligations = interface_obligations(cruise(), brake(), BRAKE_BINDINGS)
        self.assertEqual([o['name'] for o in obligations], ['interface_a_base'])
        bounds = [arg['right']['bias'] for arg in obligations[0]['ensures']['args']]
        self.assertEqual(bounds, ['-1', '1'])


@unittest.skipUnless(importlib.util.find_spec('z3'), 'optional Z3 unavailable')
class CompositionProofTests(unittest.TestCase):
    def test_whole_composition_is_equivalent_to_one_wspl_module(self):
        from skillforge.wspl_equiv import check_equivalence
        program = parse_module(COMBINED).program
        for parent in (cruise(), import_bundle(bundle_from_artifact(cruise(), provenance={'note': 'compiled stand-in'}))):
            with self.subTest(parent=parent.to_dict()['compiler']):
                d = compose_output(parent, brake(), bindings=BRAKE_BINDINGS, replace={'a': 'a'}).to_dict()
                d['source_sha256'] = hashlib.sha256(canonical_bytes(program)).hexdigest()  # relabel for the checker
                r = check_equivalence(program, ReLUArtifact(d))
                self.assertEqual(r['status'], 'EQUIVALENT', r)

    def test_brake_contracts_hold_on_the_composed_network(self):
        from skillforge.wspl_verify import verify_suite
        edited = compose_output(cruise(), brake(), bindings=BRAKE_BINDINGS, replace={'a': 'a'})
        suite = json.loads((EXAMPLES / 'brake.contracts.json').read_text(encoding='utf-8'))
        emergency = [c for c in suite['contracts'] if c['name'] in ('actuator_bound', 'emergency_brake')]
        report = verify_suite(edited, {**suite, 'artifact_sha256': edited.sha256, 'contracts': emergency})
        self.assertEqual(report['status'], 'PROVED', report)


class ValidationTests(unittest.TestCase):
    def rejected(self, message, bindings=None, replace=None, parent=None, patch=None):
        with self.assertRaisesRegex(ValueError, message):
            compose_output(parent or cruise(), patch or brake(), bindings=BRAKE_BINDINGS if bindings is None else bindings,
                           replace={'a': 'a'} if replace is None else replace)

    def test_bindings(self):
        self.rejected('every patch input', bindings={'d': 'input.d', 'v': 'input.v'})
        self.rejected('source must be', bindings={**BRAKE_BINDINGS, 'v': 'v'})
        self.rejected('source must be', bindings={**BRAKE_BINDINGS, 'v': 'parent.missing'})
        self.rejected('at most one patch input', bindings={**BRAKE_BINDINGS, 'v': 'input.d'})
        wide = compiled((EXAMPLES / 'cruise_baseline.wspl').read_text().replace('Real[0,2]', 'Real[0,3]'))
        self.rejected('not inside the patch input domain', parent=wide)
        metres = compiled('''module M {
  input d: Real[0,2] m;
  input v: Real[0,1] normalized_speed;
  output a: Real[-1,1] normalized_acceleration;
  a = clip(0.5 - v, -1, 1);
}
''')
        self.rejected("unit 'm'", parent=metres)

    def test_replacements(self):
        self.rejected('every patch output', replace={})
        self.rejected('unknown parent output', replace={'a': 'b'})
        truth_parent = compiled('''module T {
  input d: Real[0,2] normalized_distance;
  input v: Real[0,1] normalized_speed;
  output a: Truth;
  a = margin(d, 0, 2);
}
''')
        self.rejected('kind/unit differ', parent=truth_parent)


if __name__ == '__main__':
    unittest.main()
