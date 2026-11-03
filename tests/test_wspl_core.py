"""Exact scalar IR engineering examples, not scientific synthesis trials."""
import copy
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.wspl_core import ScalarProgram,rational


def example(n):return json.loads((ROOT/'examples/wspl'/n).read_text(encoding='utf-8'))
def ty(lo='0',hi='1',unit='1',kind='Real'):return dict(kind=kind,unit=unit,lower=lo,upper=hi)
def ref(n):return dict(op='ref',name=n)
def tiny(expr,inputs=None,output_type=None):
    return dict(format='wspl.scalar_ir.v0',inputs=inputs or {'x':ty()},lets=[],outputs=[dict(name='y',type=output_type or ty(),value=expr)])


class WSPLCoreTests(unittest.TestCase):
    def test_brake_demo_exact_at_boundaries_and_interior(self):
        p=ScalarProgram(example('brake_patch.core.json'))
        for d,v,base in itertools.product(['0','1/5','2/5','9/10','2'],['0','2/5','1'],['-1','0','1']):
            out=p.evaluate(dict(d=d,v=v,a_base=base))['a']
            self.assertIsInstance(out,F)
            self.assertTrue(-1<=out<=1)
            if F(d)<=F(1,5) and F(v)>=F(2,5):self.assertEqual(out,-1)
            if F(d)>=F(2,5)+F(1,2)*F(v):self.assertEqual(out,F(base))
        self.assertEqual(p.evaluate(dict(d='1/2',v='1/2',a_base='1'))['a'],F(1,4))

    def test_approach_demo_matches_exact_clipped_proportional_rule(self):
        d=example('approach_p.core.json');p=ScalarProgram(d)
        for xyz in itertools.product(['-1/2','-1/20','0','1/100','1/2'],repeat=3):
            obs={n:F(0) for n in d['inputs']};obs['aperture']=F(1,25)
            obs.update(dict(zip(['target_dx','target_dy','target_dz'],xyz)))
            out=p.evaluate(obs)
            for axis,value in zip(['x','y','z'],xyz):self.assertEqual(out['d'+axis],min(F(1),max(F(-1),20*F(value))))
            self.assertEqual(out['gripper'],-1)

    def test_graded_conjunction_is_not_boolean_idempotence(self):
        p=ScalarProgram(tiny(dict(op='and',left=ref('x'),right=ref('x')),inputs={'x':ty(kind='Truth')},output_type=ty(kind='Truth')))
        self.assertEqual(p.evaluate({'x':'3/4'})['y'],F(1,2))
        with self.assertRaises(ValueError):p.evaluate({'x':True})

    def test_unit_mismatch_duplicate_and_forward_reference(self):
        expr=dict(op='affine',unit='m',bias='0',terms=[dict(coefficient='1',value=ref('distance')),dict(coefficient='1',value=ref('speed'))])
        with self.assertRaises(ValueError):ScalarProgram(tiny(expr,inputs={'distance':ty(unit='m'),'speed':ty(unit='m_per_s')},output_type=ty('0','2','m')))
        d=example('brake_patch.core.json');d['lets'][0]['value']=ref('a')
        with self.assertRaises(ValueError):ScalarProgram(d)
        d=example('brake_patch.core.json');d['outputs'][0]['name']='d'
        with self.assertRaises(ValueError):ScalarProgram(d)

    def test_exact_constants_ranges_and_zero_normalization(self):
        self.assertEqual(rational('0.4'),F(2,5))
        for bad in (.4,float('nan'),True,'1/0'):
            with self.assertRaises(ValueError):rational(bad)
        with self.assertRaises(ValueError):ScalarProgram(tiny(dict(op='margin',value=ref('x'),lower='1',upper='1')))
        p=ScalarProgram(tiny(ref('x')))
        with self.assertRaises(ValueError):p.evaluate({'x':'2'})
        with self.assertRaises(ValueError):p.evaluate({})

    def test_unresolved_interval_obligation_is_not_assumed_true(self):
        expr=dict(op='affine',unit='1',bias='0',terms=[dict(coefficient='1',value=ref('x')),dict(coefficient='-1',value=ref('x'))])
        # Exact cancellation is valid, but this conservative interval checker
        # does not prove correlation-sensitive range facts. A solver is later.
        with self.assertRaisesRegex(ValueError,'range obligation'):ScalarProgram(tiny(expr,output_type=ty('0','0')))

    def test_snapshot_and_supported_min_max_relu(self):
        d=tiny(dict(op='max',left=dict(op='relu',value=ref('x')),right=dict(op='const',value='1/4',unit='1')))
        p=ScalarProgram(d);d['outputs'][0]['value']=dict(op='const',value='1',unit='1')
        self.assertEqual(p.evaluate({'x':'1/8'})['y'],F(1,4))
        p=ScalarProgram(tiny(dict(op='min',left=ref('x'),right=dict(op='const',value='1/2',unit='1'))))
        self.assertEqual(p.evaluate({'x':'3/4'})['y'],F(1,2))

    def test_unsupported_profile_and_real_truth_misuse_rejected(self):
        with self.assertRaises(ValueError):ScalarProgram(tiny(dict(op='sigmoid',value=ref('x'))))
        with self.assertRaises(ValueError):ScalarProgram(tiny(dict(op='not',value=ref('x'))))
        d=tiny(ref('x'));d['inputs']['x']['kind']='Bool'
        with self.assertRaises(ValueError):ScalarProgram(d)


if __name__=='__main__':unittest.main()
