import unittest,copy
import numpy as np
import torch
from skillforge.capability_data import prepare,families,canonical,orientations
from skillforge.capability_codec import prompt
from skillforge.capability_bridge import lowrank,torch_branch,validate_plan,attached
from skillforge.wsp_artifact import TensorArtifact
from skillforge.wsp_graph import node,trace_graph
from skillforge.wsp_editor import plan,apply


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=prepare()
    def test_polyomino_families(self):
        self.assertEqual([len(families(n)) for n in range(1,7)],[1,1,2,5,12,35])
        for s in families(5):
            for v in orientations(s):self.assertEqual(canonical(v),s)
    def test_whole_input_and_shape_disjoint(self):
        seen=set()
        for split in ('train','dev','test'):
            for rows in self.data[split].values():
                for r in rows:
                    key=(r['size'],tuple(r['input']));self.assertNotIn(key,seen);seen.add(key)
        f=self.data['shape_families'];self.assertFalse(set(f['train'])&set(f['novel']));self.assertFalse(set(f['train'])&set(f['larger']))
    def test_rule_and_two_steps(self):
        for split in ('train','dev','test'):
            for rows in self.data[split].values():
                for row in rows:
                    if row['task']=='copy':self.assertEqual(row['input'],row['target']);continue
                    a,b=row['input'],row['target'];color=row['color'];self.assertEqual(a.index(9),b.index(9));self.assertEqual(a.count(color),b.count(color))
                    old=sorted(divmod(i,6) for i,x in enumerate(a) if x==color);new=sorted(divmod(i,6) for i,x in enumerate(b) if x==color)
                    changes={(r1-r,c1-c) for (r,c),(r1,c1) in zip(old,new)};self.assertEqual(len(changes),1);dr,dc=changes.pop();self.assertEqual(abs(dr)+abs(dc),1)
                    mr,mc=divmod(a.index(9),6);self.assertLess(sum(abs(r-mr)+abs(c-mc) for r,c in new),sum(abs(r-mr)+abs(c-mc) for r,c in old))
                    if row['second_target'] is not None:
                        end=sorted(divmod(i,6) for i,x in enumerate(row['second_target']) if x==color)
                        self.assertEqual(end,[(r+2*dr,c+2*dc) for r,c in old])
    def test_prompt_no_target_or_solver(self):
        row=self.data['train']['move'][0];other=dict(row,target=[9]*36,second_target=[8]*36,shape='hidden')
        self.assertEqual(prompt(row),prompt(other));self.assertTrue(prompt(row).endswith('Output:'))


class BridgeTests(unittest.TestCase):
    def test_zero_and_finite_gradient(self):
        a=lowrank('0_k',rank=4);b=torch_branch(a);x=torch.randn(2,3,576)
        self.assertEqual(float(b(x).abs().max()),0);b(x).sum().backward();self.assertGreater(float(b.storage['B'].grad.norm()),0)
        self.assertEqual(a.sha256,b.artifact().sha256)
    def test_native_wspledit_equivalence(self):
        a=lowrank('0_q',rank=2);source=f'wspl 0.2; target base=artifact("base",sha256="{a.sha256}"); edit base as candidate mode=fixed {{ b[:]=1/4; }}'
        staged=plan(source,parent=a);edited=apply(staged,parent=a,expected_parent_id=a.sha256)
        arrays=a.arrays();arrays['b'][:]=.25;native=TensorArtifact.from_arrays(arrays,graph=a.to_dict()['graph'])
        self.assertEqual(native.sha256,edited.sha256);x=torch.randn(2,576)
        self.assertTrue(torch.equal(torch_branch(native)(x),torch_branch(edited)(x)))
    def test_shared_storage_and_ops(self):
        graph=dict(kind='dag',inputs={'x':3},nodes=[node('a','affine',{'x':'input:x'},params={'weight':'W','bias':'b'},attrs={'layout':'in_out'}),
          node('r','relu',{'x':'node:a'}),node('t','tanh',{'x':'node:r'}),node('s','scale',{'x':'node:t'},params={'gain':'g'}),
          node('c','concat',{'a':'node:s','b':'input:x'}),node('cut','slice',{'x':'node:c'},attrs={'start':5,'stop':2,'step':-1}),
          node('a2','affine',{'x':'node:cut'},params={'weight':'W','bias':'b'},attrs={'layout':'in_out'}),node('sum','add',{'a':'node:a2','b':'node:a'}),node('id','identity',{'x':'node:sum'})],outputs={'y':'node:id'})
        a=TensorArtifact.from_arrays(dict(W=np.eye(3,dtype=np.float32),b=np.ones(3,np.float32),g=np.array([.5],np.float32)),graph=graph);b=torch_branch(a)
        x=np.array([-.2,.3,.7],np.float32);np.testing.assert_allclose(b(torch.from_numpy(x)).detach().numpy(),trace_graph(a,{'x':x},'numpy_f32')['y'],atol=1e-6)
        self.assertEqual(len(list(b.parameters())),3)
    def test_contract_rejections(self):
        p=dict(recipe=dict(lr=.001,copy_fraction=.25,changed_cell_weight=1.,steps=128),branches={'0_q':lowrank('0_q',2).to_dict()});validate_plan(p)
        bad=copy.deepcopy(p);bad['recipe']['steps']=9999
        with self.assertRaises(ValueError):validate_plan(bad)
        bad=copy.deepcopy(p);bad['branches']['0_k']=bad['branches'].pop('0_q')
        with self.assertRaises(ValueError):validate_plan(bad)
    def test_hook_cleanup(self):
        from types import SimpleNamespace
        layer=torch.nn.Linear(576,576).requires_grad_(False);m=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=layer))]))
        bank={'0_q':torch_branch(lowrank('0_q',2))};x=torch.randn(1,576);before=layer(x).clone()
        with self.assertRaises(RuntimeError):
            with attached(m,bank):self.assertTrue(torch.equal(before,layer(x)));raise RuntimeError('exit')
        self.assertFalse(layer._forward_hooks);self.assertTrue(torch.equal(before,layer(x)))


class RunnerTests(unittest.TestCase):
    def test_freeze_all_before_final_and_independent_budgets(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from contextlib import ExitStack,nullcontext
        from types import SimpleNamespace
        from unittest.mock import patch
        from run_capability6 import run
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);out=p/'out';out.mkdir();(out/'approved_manifest.json').write_text('{}')
            (out/'source_receipt.json').write_text(json.dumps(dict(status='verified',manifest_sha256=hashlib.sha256(b'{}').hexdigest())))
            row=dict(id='fixture',size=4,task='copy',input=[0]*16,target=[0]*16)
            for name,data in [('train',{'move':[row],'copy4':[row],'copy6':[row]}),('dev',{'move':[row]}),('test',{k:[row] for k in ('familiar','novel','larger','copy4','copy6','composition')})]:
                (p/(name+'.json')).write_text(json.dumps(data))
            (p/'guide').write_text('guide');(p/'local').write_text('{"codex_cli":"synthetic"}')
            config=dict(resources={},runtime_verify='synthetic',linux_identity='synthetic',wall_guard_seconds=60,train=str(p/'train.json'),dev=str(p/'dev.json'),test=str(p/'test.json'),guide=str(p/'guide'),local_paths=str(p/'local'),catalog='synthetic',instructions='synthetic')
            tiny=dict(recipe={},branches={});call=[];chunks=[]
            def collect(*args,**kw):call.append(1);return 'FINAL',dict(thread_id=str(len(call)),usage={})
            def chunk(study,plan,train,encoded,budget,*args):chunks.append(budget['updates']);budget['updates']+=2048;return plan
            def evaluate(*args):
                self.assertEqual(len(json.loads((out/'frozen.json').read_text())),5)
                return dict(n=1,exact=1,records=[dict(id='fixture',exact=True)])
            bank=SimpleNamespace(eval=lambda:SimpleNamespace(requires_grad_=lambda flag:None))
            with ExitStack() as stack:
                for name,replacement in [('subprocess.run',lambda *a,**k:None),('model_hash',lambda m:'same'),('encode',lambda *a:([0],0)),('default_plan',lambda:tiny),('validate_plan',lambda p:dict(parameters=1)),('train_chunk',chunk),('feedback',lambda *a:{}),('collect',collect),('modules',lambda p:bank),('attached',lambda *a:nullcontext()),('evaluate',evaluate)]:stack.enter_context(patch('run_capability6.'+name,replacement))
                run(lambda:SimpleNamespace(model=SimpleNamespace(parameters=lambda:[]),tokenizer=None),out,config)
            self.assertEqual(chunks,[0,2048,4096,6144]);self.assertEqual(len(call),4)
            result=json.loads((out/'results.json').read_text());self.assertEqual(result['status'],'complete');self.assertFalse(any(v['capability_pass'] for v in result['decisions'].values()))

if __name__=='__main__':unittest.main()
