"""Compiler/interpreter/weight-runtime cross-checks; not solver certificates."""
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.wspl_core import ScalarProgram
from skillforge.wspl_compile import compile_scalar,CompilationLimit
from skillforge.wspl_artifact import ReLUArtifact


def example(n):return json.loads((ROOT/'examples/wspl'/n).read_text(encoding='utf-8'))
def ty(a,b,kind='Real'):return dict(kind=kind,unit='1',lower=a,upper=b)
def ref(n):return dict(op='ref',name=n)
def aff(terms,bias='0'):return dict(op='affine',unit='1',terms=[dict(coefficient=c,value=v) for c,v in terms],bias=bias)
def one(expr,inputs=None,out=None):return dict(format='wspl.scalar_ir.v0',inputs=inputs or {'x':ty('-2','2')},lets=[],outputs=[dict(name='out',type=out or ty('-4','4'),value=expr)])


class WSPLCompilerTests(unittest.TestCase):
    def test_brake_reproduces_explicit_two_hidden_layer_weights(self):
        a=compile_scalar(example('brake_patch.core.json'));d=a.to_dict()
        self.assertEqual(d['allocation'],dict(mode='fresh_mlp',widths=[3,4,2,1],parameters=29))
        self.assertEqual(d['layers'][0]['weights'],[['-5/2','5/4','0'],['-5/2','5/4','0'],['0','0','1'],['0','0','-1']])
        self.assertEqual(d['layers'][0]['bias'],['1','0','0','0'])
        self.assertEqual(d['layers'][1]['weights'],[['-2','2','1','-1'],['-2','2','1','-1']])
        self.assertEqual(d['layers'][1]['bias'],['1','-1'])
        self.assertEqual(d['layers'][2]['weights'],[['1','-1']]);self.assertEqual(d['layers'][2]['bias'],['-1'])
        self.assertEqual(len(d['source_map']['relu_units']),4)

    def test_brake_exact_independent_weight_execution(self):
        d=example('brake_patch.core.json');source=ScalarProgram(d);artifact=compile_scalar(d)
        for distance,speed,baseline in itertools.product(['0','1/5','2/5','1/2','9/10','2'],['0','1/3','2/5','1'],['-1','0','1']):
            obs=dict(d=distance,v=speed,a_base=baseline)
            self.assertEqual(source.evaluate(obs),artifact.evaluate(obs))

    def test_approach_independent_exact_weight_execution(self):
        d=example('approach_p.core.json');source=ScalarProgram(d);artifact=compile_scalar(d)
        for xyz in itertools.product(['-1/2','-1/20','0','1/100','1/2'],repeat=3):
            obs={n:F(0) for n in d['inputs']};obs['aperture']=F(1,25)
            obs.update(zip(['target_dx','target_dy','target_dz'],xyz))
            self.assertEqual(source.evaluate(obs),artifact.evaluate(obs))

    def test_signed_relays_deep_skip_paths_and_dead_code(self):
        d=one(dict(op='clip',lower='-2',upper='2',value=aff([('1',ref('x')),('1',ref('a')),('1',ref('c'))])),inputs={'x':ty('-2','2'),'y':ty('-2','2')},out=ty('-2','2'))
        d['lets']=[dict(name='a',type=ty('0','2'),value=dict(op='relu',value=ref('x'))),
                   dict(name='b',type=ty('0','4'),value=dict(op='relu',value=aff([('1',ref('a')),('1',ref('y'))]))),
                   dict(name='c',type=ty('0','11/2'),value=dict(op='relu',value=aff([('1',ref('b')),('-1',ref('x'))],'-1/2'))),
                   dict(name='unused',type=ty('0','101'),value=dict(op='relu',value=aff([('1',ref('x'))],'99')))]
        d['outputs'].append(dict(name='signed_y',type=ty('-2','2'),value=ref('y')))
        source=ScalarProgram(d);artifact=compile_scalar(d)
        for x,y in itertools.product(['-2','-1/3','0','1/4','2'],repeat=2):self.assertEqual(source.evaluate(dict(x=x,y=y)),artifact.evaluate(dict(x=x,y=y)))
        m=artifact.to_dict()['source_map'];self.assertTrue(m['eliminated_relu_units'])
        self.assertIn('nonnegative_relay',{x['kind'] for layer in m['allocation'] for x in layer})

    def test_operator_compositions_and_graded_semantics(self):
        inputs={'p':ty('0','1','Truth'),'q':ty('0','1','Truth')}
        for op in ['and','or','min','max']:
            d=one(dict(op=op,left=ref('p'),right=dict(op='not',value=ref('q'))),inputs=inputs,out=ty('0','1'))
            source=ScalarProgram(d);artifact=compile_scalar(d)
            for p,q in itertools.product(['0','1/3','3/4','1'],repeat=2):self.assertEqual(source.evaluate(dict(p=p,q=q)),artifact.evaluate(dict(p=p,q=q)))

    def test_serialization_hash_tampering_and_no_overwrite(self):
        a=compile_scalar(example('brake_patch.core.json'))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'weights.json';a.save(path)
            b=ReLUArtifact.load(path,a.sha256);self.assertEqual(b.sha256,a.sha256)
            with self.assertRaises(FileExistsError):a.save(path)
            d=json.loads(path.read_text());d['layers'][-1]['bias'][0]='0';path.write_text(json.dumps(d))
            with self.assertRaises(ValueError):ReLUArtifact.load(path,a.sha256)
        d=a.to_dict();d['layers'][0]['weights'][0].pop()
        with self.assertRaises(ValueError):ReLUArtifact(d)
        d=a.to_dict();d['allocation']['parameters']+=1
        with self.assertRaises(ValueError):ReLUArtifact(d)

    def test_fixed_target_and_capacity_limit_not_silent_resize(self):
        d=example('brake_patch.core.json')
        with self.assertRaises(ValueError):compile_scalar(d,target='fixed_mlp')
        with self.assertRaises(CompilationLimit):compile_scalar(d,max_parameters=1)
        with self.assertRaises(CompilationLimit):compile_scalar(d,max_layers=1)
        a=compile_scalar(one(aff([('1/3',ref('x'))]),out=ty('-1','1')))
        self.assertEqual(a.to_dict()['allocation']['widths'],[1,1])
        self.assertEqual(a.evaluate({'x':'1'})['out'],F(1,3))

    def test_cli_emits_non_executable_artifact(self):
        import os
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'brake.json'
            p=subprocess.run([sys.executable,'-m','skillforge.wspl_compile',str(ROOT/'examples/wspl/brake_patch.core.json'),'--output',str(path)],env=env,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout)['universal_verification'],'not_performed')
            self.assertEqual(ReLUArtifact.load(path).evaluate(dict(d='1/5',v='2/5',a_base='1'))['a'],-1)


try:
    import torch
except ImportError:torch=None


@unittest.skipIf(torch is None,'optional PyTorch unavailable')
class WSPLTorchEmissionTests(unittest.TestCase):
    def test_ordinary_torch_mlp_fp32_and_fp64_match_canonical_input_probes(self):
        for filename in ['brake_patch.core.json','approach_p.core.json']:
            a=compile_scalar(example(filename));d=a.to_dict()
            for dtype,tolerance in [(torch.float32,2e-6),(torch.float64,2e-12)]:
                state=torch.get_rng_state().clone();model,profile=a.to_torch(dtype)
                self.assertTrue(torch.equal(state,torch.get_rng_state()))
                self.assertEqual(profile['execution_guarantee'],'not_established')
                lower=torch.tensor([float(F(x['type']['lower'])) for x in d['inputs']],dtype=dtype)
                upper=torch.tensor([float(F(x['type']['upper'])) for x in d['inputs']],dtype=dtype)
                g=torch.Generator().manual_seed(47)
                x=lower+(upper-lower)*torch.rand(64,len(lower),dtype=dtype,generator=g)
                with torch.no_grad():ys=model(x).tolist()
                for row,output in zip(x.tolist(),ys):
                    exact=a.evaluate(dict(zip(a.input_names,[F.from_float(v) for v in row])))
                    for key,v in zip(a.output_names,output):self.assertLess(abs(v-float(exact[key])),tolerance)
                self.assertEqual(sum(p.numel() for p in model.parameters()),d['allocation']['parameters'])

    def test_nonrepresentable_weights_rejected_instead_of_silently_rounded(self):
        a=compile_scalar(one(aff([('1/3',ref('x'))]),out=ty('-1','1')))
        for dtype in [torch.float32,torch.float64]:
            with self.assertRaisesRegex(ValueError,'approximation profile'):a.to_torch(dtype)


if __name__=='__main__':unittest.main()
