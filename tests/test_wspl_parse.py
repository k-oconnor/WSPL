"""WSPL text front-end engineering checks; parsing establishes no contract by itself."""
from fractions import Fraction
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from skillforge.wspl_artifact import ReLUArtifact
from skillforge.wspl_compile import compile_scalar
from skillforge.wspl_core import ScalarProgram
from skillforge.wspl_parse import ParseError, ir_unit, parse_module

EXAMPLES = ROOT / 'examples/wspl'


def text(stem):
    return (EXAMPLES / f'{stem}.wspl').read_text(encoding='utf-8')


def committed(stem):
    return json.loads((EXAMPLES / f'{stem}.core.json').read_text(encoding='utf-8'))


def grid(document, points=5):
    axes = []
    for ty in document['inputs'].values():
        lo, hi = Fraction(ty['lower']), Fraction(ty['upper'])
        axes.append([lo + (hi - lo) * Fraction(i, points - 1) for i in range(points)])
    for values in itertools.product(*axes):
        yield dict(zip(document['inputs'], values))


def module(body, inputs='input x: Real[-1,1] normalized_x;\n  input y: Real[-1,1] normalized_y;',
           outputs='output out: Real[-10,10] normalized_out;'):
    return f'module T {{\n  {inputs}\n  {outputs}\n  {body}\n}}\n'


class ExampleTests(unittest.TestCase):
    def test_brake_example_is_the_proposal_text_verbatim(self):
        proposal = (ROOT / 'docs/wspl/PROPOSAL_v0_1_2026-11-03.md').read_text(encoding='utf-8')
        block = proposal.split('```text\n', 1)[1].split('```', 1)[0]
        example = ''.join(l for l in text('brake_patch').splitlines(keepends=True) if not l.startswith('//'))
        self.assertEqual(example, block)

    def test_brake_text_matches_committed_ir_semantics_on_grid(self):
        parsed, reference = ScalarProgram(parse_module(text('brake_patch')).program), ScalarProgram(committed('brake_patch'))
        for x in grid(committed('brake_patch')):
            self.assertEqual(parsed.evaluate(x), reference.evaluate(x), x)

    def test_brake_text_compiles_to_the_committed_layers_and_proposal_weights(self):
        layers = compile_scalar(parse_module(text('brake_patch')).program).to_dict()['layers']
        reference = ReLUArtifact.load(EXAMPLES / 'brake_patch.relu.json').to_dict()['layers']
        self.assertEqual(layers, reference)
        first = list(zip(map(tuple, layers[0]['weights']), layers[0]['bias']))
        # Proposal: h1 = ReLU(1 + 1.25v - 2.5d), h2 = ReLU(1.25v - 2.5d), relays p, n for a_base.
        for row in [(('-5/2', '5/4', '0'), '1'), (('-5/2', '5/4', '0'), '0'),
                    (('0', '0', '1'), '0'), (('0', '0', '-1'), '0')]:
            self.assertIn(row, first)

    def test_approach_text_reproduces_committed_ir_and_contract_suite(self):
        parsed = parse_module(text('approach_p'))
        self.assertEqual(parsed.program, committed('approach_p'))
        artifact = ReLUArtifact.load(EXAMPLES / 'approach_p.relu.json')
        expected = json.loads((EXAMPLES / 'approach.contracts.json').read_text(encoding='utf-8'))
        self.assertEqual(parsed.contract_suite(artifact), expected)

    def test_report_records_unit_labels_contract_kinds_and_identities(self):
        parsed = parse_module(text('brake_patch'))
        r = parsed.report
        self.assertEqual(r['module'], 'BrakePatch')
        self.assertEqual(r['unit_labels']['d'], 'normalized_distance')
        self.assertEqual(r['contract_kinds'], {'actuator_bound': 'ensures', 'emergency_brake': 'ensures',
                                               'clear_path': 'preserves'})
        self.assertIn('not implemented', r['preserves_semantics'])
        self.assertEqual(r['ir_sha256'], compile_scalar(parsed.program).to_dict()['source_sha256'])
        self.assertEqual(len(r['text_sha256']), 64)

    def test_contract_split_and_binding(self):
        parsed = parse_module(text('brake_patch'))
        by_name = {c['name']: c for c in parsed.contracts}
        self.assertEqual(by_name['actuator_bound']['assumptions'], {'op': 'true'})
        self.assertEqual(by_name['emergency_brake']['assumptions']['op'], 'and')
        self.assertEqual(by_name['emergency_brake']['ensures']['op'], 'eq')
        with self.assertRaisesRegex(ValueError, 'not compiled from this module'):
            parsed.contract_suite(ReLUArtifact.load(EXAMPLES / 'approach_p.relu.json'))


@unittest.skipUnless(importlib.util.find_spec('z3'), 'optional Z3 unavailable')
class ContractVerificationTests(unittest.TestCase):
    def setUp(self):
        from skillforge.wspl_verify import verify_suite
        self.verify_suite = verify_suite
        self.parsed = parse_module(text('brake_patch'))

    def test_parsed_brake_contracts_prove_on_compiled_artifact(self):
        artifact = compile_scalar(self.parsed.program)
        report = self.verify_suite(artifact, self.parsed.contract_suite(artifact))
        self.assertEqual(report['status'], 'PROVED', report)
        self.assertTrue(all(r['nonvacuous'] for r in report['results']))

    def test_parsed_brake_contracts_refute_known_regressions(self):
        for mutate in ('gain', 'columns'):
            with self.subTest(mutation=mutate):
                d = compile_scalar(self.parsed.program).to_dict()
                if mutate == 'gain':
                    for row in d['layers'][1]['weights']:
                        row[0], row[1] = '-1', '1'
                else:
                    for row in d['layers'][0]['weights']:
                        row[0], row[1] = row[1], row[0]
                artifact = ReLUArtifact(d)
                report = self.verify_suite(artifact, self.parsed.contract_suite(artifact))
                statuses = {r['name']: r['status'] for r in report['results']}
                self.assertEqual(statuses['emergency_brake'], 'REFUTED', report)


class LoweringTests(unittest.TestCase):
    def test_equivalent_constant_spellings_give_identical_ir(self):
        spellings = ['0.4', '0.40', '2/5', '4/10', '(0.2 + 0.2)', '0.8/2']
        programs = [parse_module(module(f'out = {c}*x;')).program for c in spellings]
        self.assertTrue(all(p == programs[0] for p in programs))
        self.assertEqual(programs[0]['outputs'][0]['value']['terms'][0]['coefficient'], '2/5')

    def test_affine_terms_merge_and_single_unit_terms_are_bare(self):
        p = parse_module(module('out = 2*x - x + 0*y;')).program
        self.assertEqual(p['outputs'][0]['value'], {'op': 'ref', 'name': 'x'})
        p = parse_module(module('out = (x + y)/2 + x;')).program
        value = p['outputs'][0]['value']
        self.assertEqual([t['coefficient'] for t in value['terms']], ['3/2', '1/2'])

    def test_implies_expression_is_rewritten_exactly(self):
        source = module('out = implies(p, q);', inputs='input p: Truth;\n  input q: Truth;',
                        outputs='output out: Truth;')
        parsed = parse_module(source)
        self.assertEqual(len(parsed.report['rewrites']), 1)
        program = ScalarProgram(parsed.program)
        for x in grid(parsed.program, points=7):
            self.assertEqual(program.evaluate(x)['out'], min(Fraction(1), 1 - x['p'] + x['q']))

    def test_every_operator_parses_and_matches_hand_semantics(self):
        source = module(
            'let r: Real[0,1] = relu(x);\n  let m: Real[-1,1] = min(x, y);\n  let n: Real[-1,1] = max(y, -0.5);\n'
            '  let g: Truth = margin(x, -0.5, 0.5);\n  let c: Truth = and(p, q);\n  let o: Truth = or(p, not(q));\n'
            '  out = r + 0.5*m - n + c + o/3 - 0.4*g + 1/7;',
            inputs='input x: Real[-1,1] normalized_x;\n  input y: Real[-1,1] normalized_y;\n'
                   '  input p: Truth;\n  input q: Truth;')
        program = ScalarProgram(parse_module(source).program)
        for v in grid({'inputs': {k: {'lower': lo, 'upper': '1'} for k, lo in
                                  [('x', '-1'), ('y', '-1'), ('p', '0'), ('q', '0')]}}, points=4):
            x, y, p, q = v['x'], v['y'], v['p'], v['q']
            expected = (max(0, x) + Fraction(1, 2) * min(x, y) - max(y, Fraction(-1, 2)) + max(0, p + q - 1)
                        + min(Fraction(1), p + 1 - q) / 3 - Fraction(2, 5) * min(1, max(0, x + Fraction(1, 2))) + Fraction(1, 7))
            self.assertEqual(program.evaluate(v)['out'], expected, v)

    def test_unit_rule(self):
        self.assertEqual(ir_unit('normalized_speed'), '1')
        self.assertEqual(ir_unit('m_per_s'), 'm_per_s')


class RejectionTests(unittest.TestCase):
    def assert_rejected(self, source, message, at=None):
        """`at` is a snippet whose first line in `source` the error must point to."""
        with self.assertRaises(ParseError) as caught:
            parse_module(source)
        self.assertIn(message, str(caught.exception))
        if at is not None:
            expected = next(i for i, l in enumerate(source.splitlines(), 1) if at in l)
            self.assertEqual(caught.exception.position[0], expected, str(caught.exception))

    def test_core_pwa_boundaries(self):
        self.assert_rejected(module('out = x*y;'), 'product of two varying values', 'x*y')
        self.assert_rejected(module('out = x/y;'), 'division by a varying value', 'x/y')
        self.assert_rejected(module('out = x/0;'), 'division by zero', 'x/0')
        self.assert_rejected(module('out = tanh(x);'), 'unknown or unsupported function', 'tanh')
        self.assert_rejected(module('requires domain: forall inputs: x <= 1;'), 'outside the Core PWA', 'requires')
        self.assert_rejected(module('out = clip(x, y, 1);'), 'constant required', 'clip')
        self.assert_rejected(module('out = clip(x, 1, 0);'), 'invalid clip interval', 'clip')
        self.assert_rejected(module('out = min(x);'), 'takes 2 arguments', 'min')

    def test_lexical_errors(self):
        self.assert_rejected(module('out = 1e3*x;'), "expected ';'", '1e3')
        self.assert_rejected(module('out = x # y;'), 'unexpected character', '#')
        self.assert_rejected(module('ensures c: forall inputs: x != 1;'), 'unexpected character', '!=')
        self.assert_rejected(module('out = x;') + 'extra', 'text after module end')
        self.assert_rejected('module T {\n  input x: Real[0,1] normalized_x;', "expected '}'")

    def test_units_and_types(self):
        self.assert_rejected(module('out = x + y;', inputs='input x: Real[0,1] m;\n  input y: Real[0,1] m_per_s;'),
                             'normalize explicitly', 'x + y')
        self.assert_rejected(module('out = x;', inputs='input x: Real[0,1];'), 'needs a unit', 'input x')
        self.assert_rejected(module('out = x;', inputs='input x: Real m;'), 'explicit [lower, upper] bounds', 'input x')
        self.assert_rejected(module('out = x;', inputs='input x: Truth m;'), 'no unit label', 'input x')
        self.assert_rejected(module('out = x;', inputs='input X: Real[0,1] m;'), 'must match', 'input X')
        self.assert_rejected(module('let t: Truth = x;\n  out = t;', inputs='input x: Real[0,1] m;'),
                             "expression has unit 'm'", 'let t')
        self.assert_rejected(module('let w: Real[0,1] = 3*x;\n  out = w;'), 'range obligation')

    def test_declarations_and_scope(self):
        self.assert_rejected(module(''), "output 'out' is never assigned", 'output out')
        self.assert_rejected(module('out = x;\n  out = y;'), 'assigned twice', 'out = y')
        self.assert_rejected(module('x = y;\n  out = x;'), 'not a declared output', 'x = y')
        self.assert_rejected(module('let a: Real[-1,1] = b;\n  let b: Real[-1,1] = x;\n  out = a;'),
                             'unknown or forward reference', 'let a')
        self.assert_rejected(module('out = x;', inputs='input x: Real[0,1] m;\n  input x: Real[0,2] m;'),
                             'duplicate declaration', 'Real[0,2]')
        self.assert_rejected(module('let t: Real[-1,1] = x;\n  out = t;\n  ensures c: forall inputs: t <= 1;'),
                             'ports only', 'ensures c')
        self.assert_rejected(module('out = x;\n  ensures c: forall inputs: out <= 10;\n  ensures c: forall inputs: true;'),
                             'duplicate contract', 'true;')
        self.assert_rejected(module('let relu: Real[0,1] = x;\n  out = x;'), 'reserved', 'let relu')

    def test_deep_nesting_is_a_parse_error_not_a_crash(self):
        self.assert_rejected(module('out = ' + '(' * 200 + 'x' + ')' * 200 + ';'), 'nesting limit')
        self.assert_rejected(module('out = x;\n  ensures c: forall inputs: ' + 'not ' * 200 + 'true;'), 'nesting limit')


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'skillforge.wspl_parse', *map(str, args)],
                              capture_output=True, text=True, cwd=ROOT,
                              env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})

    def test_cli_writes_ir_report_and_bound_contracts_exclusively(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            result = self.run_cli(EXAMPLES / 'approach_p.wspl', '--ir-output', tmp / 'ir.json',
                                  '--report-output', tmp / 'report.json',
                                  '--artifact', EXAMPLES / 'approach_p.relu.json', '--contracts-output', tmp / 'c.json')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((tmp / 'ir.json').read_text()), committed('approach_p'))
            self.assertEqual(json.loads((tmp / 'c.json').read_text()),
                             json.loads((EXAMPLES / 'approach.contracts.json').read_text()))
            self.assertEqual(json.loads((tmp / 'report.json').read_text())['module'], 'ApproachP')
            again = self.run_cli(EXAMPLES / 'approach_p.wspl', '--ir-output', tmp / 'ir.json')
            self.assertEqual(again.returncode, 1)

    def test_cli_rejects_unbound_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            result = self.run_cli(EXAMPLES / 'approach_p.wspl', '--ir-output', tmp / 'ir.json',
                                  '--artifact', EXAMPLES / 'brake_patch.relu.json', '--contracts-output', tmp / 'c.json')
            self.assertEqual(result.returncode, 1)
            self.assertIn('not compiled from this module', result.stderr)
            self.assertFalse((tmp / 'c.json').exists())


if __name__ == '__main__':
    unittest.main()
