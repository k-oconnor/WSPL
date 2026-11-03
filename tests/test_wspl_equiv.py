"""Solver-backed source/artifact equivalence engineering fixtures, not physical guarantees."""
from fractions import Fraction
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.wspl_artifact import ReLUArtifact
from skillforge.wspl_compile import compile_scalar
from skillforge.wspl_core import ScalarProgram
from skillforge.wspl_equiv import check_equivalence

EXAMPLES=ROOT/'examples/wspl'


def source(stem):return json.loads((EXAMPLES/f'{stem}.core.json').read_text(encoding='utf-8'))
def artifact(stem):return ReLUArtifact.load(EXAMPLES/f'{stem}.relu.json')


def grid(document,points=5):
    axes=[]
    for ty in document['inputs'].values():
        lo,hi=Fraction(ty['lower']),Fraction(ty['upper'])
        axes.append([lo+(hi-lo)*Fraction(i,points-1) for i in range(points)])
    for values in itertools.product(*axes):yield dict(zip(document['inputs'],values))


@unittest.skipUnless(importlib.util.find_spec('z3'),'optional Z3 unavailable')
class EquivalenceTests(unittest.TestCase):
    def assert_replayed_difference(self,document,a,r):
        self.assertEqual(r['status'],'DIFFERENT',r);self.assertTrue(r['witness_validated'])
        x=r['counterexample']['inputs'];expected=ScalarProgram(document).evaluate(x);actual=a.evaluate(x)
        self.assertTrue(r['counterexample']['differing_outputs'])
        for key in r['counterexample']['differing_outputs']:self.assertNotEqual(expected[key],actual[key])

    def test_committed_examples_are_equivalent_to_their_sources(self):
        for stem in ('brake_patch','approach_p'):
            with self.subTest(stem=stem):
                r=check_equivalence(source(stem),artifact(stem))
                self.assertEqual(r['status'],'EQUIVALENT',r);self.assertEqual(r['difference_query'],'unsat')
                self.assertEqual(r['numerical_execution'],'not_established')
                self.assertEqual(r['physical_trajectory'],'not_established')
                self.assertFalse(r['independently_checked_proof_trace'])

    def test_fresh_compilation_round_trip_is_equivalent(self):
        for stem in ('brake_patch','approach_p'):
            with self.subTest(stem=stem):
                d=source(stem);self.assertEqual(check_equivalence(d,compile_scalar(d))['status'],'EQUIVALENT')

    def test_reduced_braking_gain_is_different_with_replayed_witness(self):
        d=artifact('brake_patch').to_dict()
        for row in d['layers'][1]['weights']:row[0]='-1';row[1]='1'
        a=ReLUArtifact(d);self.assert_replayed_difference(source('brake_patch'),a,check_equivalence(source('brake_patch'),a))

    def test_swapped_input_columns_are_different(self):
        d=artifact('brake_patch').to_dict()
        for row in d['layers'][0]['weights']:row[0],row[1]=row[1],row[0]
        a=ReLUArtifact(d);self.assert_replayed_difference(source('brake_patch'),a,check_equivalence(source('brake_patch'),a))

    def test_tiny_readout_bias_change_is_detected_exactly(self):
        d=artifact('brake_patch').to_dict();d['layers'][-1]['bias'][0]=str(Fraction(d['layers'][-1]['bias'][0])+Fraction(1,10**6))
        a=ReLUArtifact(d);self.assert_replayed_difference(source('brake_patch'),a,check_equivalence(source('brake_patch'),a))

    def test_single_coefficient_mutations_are_classified_consistently(self):
        document=source('brake_patch');base=artifact('brake_patch').to_dict();program=ScalarProgram(document)
        locations=[(i,'weights',j,k) for i,layer in enumerate(base['layers']) for j,row in enumerate(layer['weights']) for k in range(len(row))]
        locations+=[(i,'bias',j,None) for i,layer in enumerate(base['layers']) for j in range(len(layer['bias']))]
        self.assertEqual(len(locations),base['allocation']['parameters'])
        statuses=set()
        for i,field,j,k in locations:
            d=artifact('brake_patch').to_dict();values=d['layers'][i][field]
            if k is None:values[j]=str(Fraction(values[j])+1)
            else:values[j][k]=str(Fraction(values[j][k])+1)
            a=ReLUArtifact(d);r=check_equivalence(document,a);statuses.add(r['status'])
            with self.subTest(layer=i,field=field,row=j,column=k):
                self.assertIn(r['status'],('EQUIVALENT','DIFFERENT'),r)
                if r['status']=='DIFFERENT':self.assert_replayed_difference(document,a,r)
                else:
                    for x in grid(document):self.assertEqual(program.evaluate(x),a.evaluate(x))
        self.assertIn('DIFFERENT',statuses)

    def test_source_edit_breaks_identity_binding(self):
        d=source('brake_patch');d['lets'][0]['value']['value']['bias']='1/2'
        r=check_equivalence(d,artifact('brake_patch'))
        self.assertEqual(r['status'],'INVALID');self.assertIn('not bound',r['reason'])

    def test_port_name_and_bound_mismatches_are_invalid(self):
        d=artifact('brake_patch').to_dict();d['outputs'][0]['name']='accel'
        self.assertEqual(check_equivalence(source('brake_patch'),ReLUArtifact(d))['status'],'INVALID')
        d=artifact('brake_patch').to_dict();d['inputs'][0]['type']['upper']='3'
        self.assertEqual(check_equivalence(source('brake_patch'),ReLUArtifact(d))['status'],'INVALID')

    def test_budget_and_capacity_limits_are_unknown(self):
        self.assertEqual(check_equivalence(source('brake_patch'),artifact('brake_patch'),timeout_ms=0)['status'],'UNKNOWN')
        self.assertEqual(check_equivalence(source('brake_patch'),artifact('brake_patch'),max_parameters=1)['status'],'UNKNOWN')

    def test_malformed_source_is_invalid(self):
        self.assertEqual(check_equivalence({'format':'wspl.scalar_ir.v0'},artifact('brake_patch'))['status'],'INVALID')


def every_operator_program():
    real=lambda lo,hi:dict(kind='Real',unit='1',lower=lo,upper=hi)
    truth=dict(kind='Truth',unit='1',lower='0',upper='1')
    ref=lambda n:dict(op='ref',name=n)
    let=lambda n,ty,v:dict(name=n,type=ty,value=v)
    return {'format':'wspl.scalar_ir.v0',
            'inputs':{'x':real('-1','1'),'y':real('-1','1'),'p':truth,'q':truth},
            'lets':[let('r',real('0','1'),dict(op='relu',value=ref('x'))),
                    let('m',real('-1','1'),dict(op='min',left=ref('x'),right=ref('y'))),
                    let('n',real('-1','1'),dict(op='max',left=ref('y'),right=dict(op='const',value='-1/2',unit='1'))),
                    let('c',truth,dict(op='and',left=ref('p'),right=ref('q'))),
                    let('o',truth,dict(op='or',left=ref('p'),right=dict(op='not',value=ref('q')))),
                    let('g',truth,dict(op='margin',value=ref('x'),lower='-1/2',upper='1/2'))],
            'outputs':[let('out',real('-3','4'),dict(op='affine',bias='1/7',unit='1',terms=[
                           dict(coefficient='1',value=ref('r')),dict(coefficient='1/2',value=ref('m')),
                           dict(coefficient='-1',value=ref('n')),dict(coefficient='1',value=ref('c')),
                           dict(coefficient='1/3',value=ref('o')),dict(coefficient='-2/5',value=ref('g'))])),
                       let('bounded',real('-1/2','1/2'),dict(op='clip',lower='-1/2',upper='1/2',value=dict(
                           op='affine',bias='0',unit='1',terms=[dict(coefficient='1',value=ref('x')),dict(coefficient='1',value=ref('y'))])))]}


@unittest.skipUnless(importlib.util.find_spec('z3'),'optional Z3 unavailable')
class OperatorCoverageTests(unittest.TestCase):
    """Exercises every scalar IR operator, which the committed examples do not."""

    def test_source_encoder_matches_exact_interpreter_on_grid(self):
        import z3
        from skillforge.wspl_equiv import _encode_source
        document=every_operator_program();program=ScalarProgram(document)
        for point in grid(document,points=4):
            xs={k:z3.Real('input_'+k) for k in document['inputs']};constraints=[]
            outputs=_encode_source(z3,document,xs,constraints)
            solver=z3.Solver();solver.add(*constraints,*(xs[k]==z3.RealVal(str(v)) for k,v in point.items()))
            self.assertEqual(solver.check(),z3.sat)
            model=solver.model();expected=program.evaluate(point)
            for key,term in zip(expected,outputs):
                v=model.eval(term,model_completion=True)
                self.assertEqual(Fraction(v.numerator_as_long(),v.denominator_as_long()),expected[key],(point,key))

    def test_compiled_every_operator_program_is_equivalent(self):
        d=every_operator_program();r=check_equivalence(d,compile_scalar(d))
        self.assertEqual(r['status'],'EQUIVALENT',r)

    def test_artifacts_compiled_from_wrong_operators_are_different(self):
        original=every_operator_program();source_sha=check_equivalence(original,compile_scalar(original))['source_sha256']
        swaps={'min':'max','max':'min','and':'or','or':'and'}
        for index,declaration in enumerate(original['lets']):
            op=declaration['value']['op']
            if op not in swaps:continue
            with self.subTest(let=declaration['name'],swap=f'{op}->{swaps[op]}'):
                wrong=every_operator_program();wrong['lets'][index]['value']['op']=swaps[op]
                d=compile_scalar(wrong).to_dict();d['source_sha256']=source_sha  # mislabelled as compiled from the original
                a=ReLUArtifact(d);r=check_equivalence(original,a)
                self.assertEqual(r['status'],'DIFFERENT',r);self.assertTrue(r['witness_validated'])
                x=r['counterexample']['inputs']
                self.assertNotEqual(ScalarProgram(original).evaluate(x),a.evaluate(x))


class IndependenceTests(unittest.TestCase):
    def test_checker_does_not_import_the_compiler(self):
        text=(ROOT/'src/skillforge/wspl_equiv.py').read_text(encoding='utf-8')
        self.assertNotIn('wspl_compile',text)


if __name__=='__main__':unittest.main()
