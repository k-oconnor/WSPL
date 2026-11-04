"""Construction/routing tests only; no model requests or scored game episodes."""
from fractions import Fraction as Q
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.breakout_matched import MatchedBreakoutPolicy,code_body,wspl_body,INPUTS,NODE_LIMIT
from skillforge.wspl_core import ScalarProgram
from skillforge.wspl_equiv import check_equivalence
from run_breakout_matched import ORDER,contrasts

CODE='h = relu(x003 - 0.5)\nreturn [0, -h, 0, h, 0, 0]'
WSPL='let h: Real[0,0.5] normalized_value = relu(x003 - 0.5);\na0=0; a1=-h; a2=0; a3=h; a4=0; a5=0;'
WEIGHTS='W1[3,0]=1\nb1[0]=-0.5\nW2[0,0]=1\nW3[0,1]=-1\nW3[0,3]=1'


def load(arm,text):
    p=MatchedBreakoutPolicy();p.load_answer(arm,'```'+{'weights':'edits','code':'python','wspl':'wspl'}[arm]+'\n'+text+'\n```');return p


class MatchedTests(unittest.TestCase):
    def test_all_frontends_same_target_and_logits(self):
        policies=[load(a,t) for a,t in zip(('weights','code','wspl'),(WEIGHTS,CODE,WSPL))]
        for p in policies:
            self.assertEqual(sum(a.size for a in p.weights.values()),20486)
            self.assertEqual(p.artifact.to_dict()['allocation']['widths'],[600,32,32,6])
            for v in (0,.25,.5,.75,1):
                x=np.zeros(600,dtype=np.float32);x[3]=v
                expected=np.zeros(6,dtype=np.float32);expected[1]=-max(0,v-.5);expected[3]=max(0,v-.5)
                np.testing.assert_array_equal(p.logits(x),expected)
                inp={n:'0' for n in INPUTS};inp['x003']=str(Q(v))
                exact=p.artifact.evaluate(inp)
                self.assertEqual([float(exact['a'+str(i)]) for i in range(6)],expected.tolist())

    def test_signed_affine_padding_and_constant_heads(self):
        p=load('code','return [x000-x001, -x000, 0, 2, -2, x001]')
        x=np.zeros(600,dtype=np.float32);x[0]=.75;x[1]=.25
        np.testing.assert_array_equal(p.logits(x),[.5,-.75,0,2,-2,.25])

    def test_full_two_by_32_capacity_matches_native_weights(self):
        code=[];wspl=[]
        for i in range(32):
            code.append(f'h{i}=relu(x{i:03}-0.5)')
            wspl.append(f'let h{i}: Real[0,0.5] normalized_value=relu(x{i:03}-0.5);')
        for i in range(32):
            code.append(f'g{i}=relu(h{i}-0.25)')
            wspl.append(f'let g{i}: Real[0,0.25] normalized_value=relu(h{i}-0.25);')
        outputs=[];matrix=[]
        for j in range(6):
            expr='+'.join(f'{"0.5" if (i+j)%2 else "-0.5"}*g{i}' for i in range(32))
            outputs.append(expr);wspl.append(f'a{j}={expr};')
        code.append('return ['+','.join(outputs)+']')
        native=[]
        for i in range(32):
            native.extend((f'W1[{i},{i}]=1',f'W2[{i},{i}]=1'))
            matrix.append([.5 if (i+j)%2 else -.5 for j in range(6)])
        native.extend(('b1[:]=-0.5','b2[:]=-0.25','W3[:,:]='+repr(matrix)))
        policies=[load('code','\n'.join(code)),load('wspl','\n'.join(wspl)),load('weights','\n'.join(native))]
        x=(np.arange(600,dtype=np.float32)%5)/4
        for p in policies:
            self.assertEqual(p.artifact.to_dict()['allocation']['widths'],[600,32,32,6])
            np.testing.assert_array_equal(p.logits(x),policies[-1].logits(x))

    def test_deeper_and_wider_compilation_rejected(self):
        for code in ('a=relu(x000)\nb=relu(a-1)\nc=relu(b-1)\nreturn [c,0,0,0,0,0]',
                     '\n'.join(f'h{i}=relu(x{i:03})' for i in range(33))+'\nreturn ['+'+'.join(f'h{i}' for i in range(33))+',0,0,0,0,0]'):
            with self.assertRaises(ValueError):load('code',code)

    def test_code_subset_does_not_execute_arbitrary_python(self):
        for text in ('import os\nreturn [0,0,0,0,0,0]',
                     'return [__import__("os").system("ignored"),0,0,0,0,0]',
                     'a=x000*x001\nreturn [a,0,0,0,0,0]',
                     'for i in range(2):\n a=0\nreturn [0,0,0,0,0,0]',
                     'return [x[0],0,0,0,0,0]','a=x000\na=0\nreturn [a,0,0,0,0,0]',
                     'relu=x000\nreturn [relu,0,0,0,0,0]'):
            with self.assertRaises(ValueError):load('code',text)

    def test_wspl_interface_and_external_requirements(self):
        for body in (WSPL+'\ninput extra:Truth;',WSPL+'\nensures mine: forall inputs: true;',
                     WSPL.replace('a5=0;',''),WSPL+'\n} module other {'):
            with self.assertRaises(ValueError):load('wspl',body)

    def test_common_rounding_is_visible_and_can_break_source_fidelity(self):
        p=load('code','return [0.1,0,0,0,0,0]')
        self.assertEqual(float(p.logits(np.zeros(600,dtype=np.float32))[0]),float(np.float32(.1)))
        self.assertGreater(p.artifact.to_dict()['source_map']['coefficients_changed'],0)
        result=check_equivalence(p.program,p.artifact,max_parameters=20486,max_source_nodes=NODE_LIMIT)
        self.assertEqual(result['status'],'DIFFERENT')

    def test_source_fidelity_on_dyadic_example(self):
        p=load('wspl',WSPL)
        result=check_equivalence(p.program,p.artifact,max_parameters=20486,max_source_nodes=NODE_LIMIT)
        self.assertEqual(result['status'],'EQUIVALENT')

    def test_large_affine_body_budget_and_unchanged_default_guard(self):
        code='h=relu('+'+'.join('2*'+n for n in INPUTS)+')\nreturn [h,0,0,0,0,0]'
        program=code_body(code)
        with self.assertRaises(ValueError):ScalarProgram(program)
        ScalarProgram(program,max_nodes=NODE_LIMIT)
        p=load('code',code)
        self.assertEqual(p.logits(np.ones(600,dtype=np.float32))[0],1200)
        for limit in (0,True,100001):
            with self.assertRaises(ValueError):ScalarProgram(program,max_nodes=limit)

    def test_oversized_missing_and_multiple_final_blocks(self):
        p=MatchedBreakoutPolicy()
        with self.assertRaises(ValueError):p.load_answer('wspl','missing block')
        with self.assertRaises(ValueError):MatchedBreakoutPolicy().load_answer('code','x'*1000001)
        p=load('code','return [0,0,0,0,0,0]\n```\n```python\nreturn [1,0,0,0,0,0]')
        self.assertEqual(p.logits(np.zeros(600,dtype=np.float32))[0],1)

    def test_balanced_order_and_denominator_decision(self):
        self.assertEqual(len(set(ORDER)),9)
        for start in (0,3,6):self.assertEqual({a for a,i in ORDER[start:start+3]},{'weights','code','wspl'})
        trials=[dict(arm=a,draw=i,status='complete',summary={'mean':200}) for a,i in ORDER]
        self.assertEqual(contrasts(trials)['internal_feasibility_decision'],'go')
        trials[2].update(status='invalid_proposal',summary=None)
        c=contrasts(trials);self.assertAlmostEqual(c['arms']['wspl']['failure_adjusted_mean'],400/3)
        self.assertEqual(c['internal_feasibility_decision'],'no_go_for_this_configuration')
        trials[0]['status']='collection_failed'
        self.assertEqual(contrasts(trials)['internal_feasibility_decision'],'unresolved')
