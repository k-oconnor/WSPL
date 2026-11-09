import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from unittest.mock import patch
from contextlib import nullcontext
from skillforge.joint_attention import validate,weight_edit,metrics
from skillforge.joint_family import family
from run_joint_attention import session


def bank():
    b=dict(K=np.zeros((6,3,64),np.float32),V=np.zeros((6,3,64),np.float32),alpha=np.zeros((6,3),np.float32),
           K_scale=np.ones((6,3),np.float32),V_scale=np.ones((6,3),np.float32),W27=np.eye(4,dtype=np.float32))
    b['alpha'][2,0]=.75;b['original_W27']=b['W27'].copy();return b


class JointTests(unittest.TestCase):
    def test_contract(self):
        r=bank();b=bank();b['K'][2,0,0]=1;b['V'][2,0,1]=2;b['W27']*=1.01;validate(b,r)
        for name,edit in [('slot',lambda b:b['K'].__setitem__((0,0,0),1)),('weight',lambda b:b['W27'].__imul__(1.2)),
                          ('readonly',lambda b:b['original_W27'].__imul__(1.01)),('norm',lambda b:b['V'].__setitem__((2,0,0),11))]:
            b=bank();edit(b)
            with self.subTest(name=name),self.assertRaises(ValueError):validate(b,r)

    def test_weight_restore_on_exception(self):
        import torch
        layer=SimpleNamespace(self_attn=SimpleNamespace(o_proj=torch.nn.Linear(4,4,bias=False)))
        model=SimpleNamespace(model=SimpleNamespace(layers=[layer]*28));old=layer.self_attn.o_proj.weight.detach().clone()
        with self.assertRaises(RuntimeError),weight_edit(model,bank()):raise RuntimeError('fixture')
        self.assertTrue(torch.equal(layer.self_attn.o_proj.weight,old))

    def test_last_accepted_and_invalid_retention(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);n=[0];evaluated=[]
            def collect(folder,text):folder.mkdir();n[0]+=1;return ('FINAL' if n[0]==3 else '```python\npass\n```'),{}
            def compile(source,current,folder):
                b=bank();b['W27']*=1.01 if n[0]==1 else 2
                path=folder/'candidate.npz';np.savez_compressed(path,**b);return path,dict(exit_code=0)
            result=session(bank(),p/'author',{},'',collect,compile,lambda b:evaluated.append(1) or {},lambda *a:None)
            self.assertEqual(len(evaluated),1);np.testing.assert_array_equal(result['W27'],bank()['W27']*1.01)

    def test_family_and_metric_accounting(self):
        a=family(0);b=family(1)+family(2)
        self.assertFalse({r['id'] for r in a}&{r['id'] for r in b})
        # Situation questions repeat by concept; answer contexts differ. Do not
        # call this question-string-disjoint or independent linguistic evidence.
        self.assertEqual(len(a),8);self.assertEqual(len(b),16)
        base=[dict(id=r['id'],correct=False) for r in a];new=[dict(id=r['id'],correct=True) for r in a]
        self.assertEqual(len(metrics(new,base,a)['corrected']),8)

    def test_runner_freezes_before_reserved_and_runs_ablations(self):
        import hashlib
        from run_joint_attention import run
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'approved_manifest.json').write_text('{}');h=hashlib.sha256(b'{}').hexdigest()
            (p/'source_receipt.json').write_text(json.dumps(dict(status='verified',manifest_sha256=h)))
            rows=family(0)[:1];(p/'acq.json').write_text(json.dumps(rows));reserved=p/'reserved.json';reserved.write_text(json.dumps(rows))
            (p/'guide.md').write_text('fixture');(p/'local.json').write_text(json.dumps(dict(codex_cli='fake')))
            np.savez(p/'parent.npz',**bank());read=Path.read_text;modes=[]
            def guarded(path,*args,**kwargs):
                if path==reserved:self.assertTrue((p/'frozen.json').exists())
                return read(path,*args,**kwargs)
            def collector(cli,catalog,instructions,folder,text,timeout,heartbeat):folder.mkdir();return 'FINAL',dict(thread_id='fixture')
            def score(study,row,b,mode):modes.append(mode);return dict(id=row['id'],correct=True,answer_logprobs=[-1,-2])
            baseline=[dict(id=rows[0]['id'],correct=True)]
            config=dict(resources={},runtime_verify='fake',linux_identity='fake',wall_guard_seconds=100,model_sha256='frozen',parent_bank=str(p/'parent.npz'),acquisition=str(p/'acq.json'),reserved=str(reserved),guide=str(p/'guide.md'),local_paths=str(p/'local.json'),catalog='fake',instructions='fake')
            with patch('run_joint_attention.subprocess.run'),patch('run_joint_attention.model_hash',return_value='frozen'),patch('run_joint_attention.make_reference',return_value=(bank(),baseline,baseline)),patch('run_joint_attention.collect',collector),patch('run_joint_attention.weight_edit',lambda *a:nullcontext()),patch('run_joint_attention.score_boundary',score),patch.object(Path,'read_text',guarded):
                run(lambda:SimpleNamespace(model=None),p,config)
            self.assertEqual(len(modes),5);self.assertEqual(json.loads((p/'results.json').read_text())['status'],'complete')


if __name__=='__main__':unittest.main()
