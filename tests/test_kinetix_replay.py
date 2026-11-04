import unittest
import json
import sys
import tempfile
import os
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import numpy as np
from skillforge.kinetix_replay import parse_parent,reconstruct_native,sample_action,ReplayStudy,SHAPES

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'data/csail_kinetix_reference_v1/exp4_fix_car'

class KinetixTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('SKILLFORGE_KINETIX_FIXTURE')=='1','explicit isolated Kinetix fixture only')
    def test_three_native_zero_action_ticks(self):
        import jax
        import jax.numpy as jnp
        from skillforge.kinetix_replay import make_environment
        e,s,p=make_environment('/mnt/c/Users/kevin/skillforge-runtime/Kinetix/kinetix/levels/l/h2_car_ramp.json')
        o,s=e.reset(jax.random.PRNGKey(0),p,override_reset_state=s)
        self.assertEqual(o.shape,(679,))
        np.testing.assert_allclose(np.asarray(o)[52:54],[2.8,.725],atol=1e-6)
        np.testing.assert_allclose(np.asarray(o)[234:236],[2.5375,.735],atol=1e-6)
        for i in range(3):
            o,s,_,_,_=e.step(jax.random.PRNGKey(i),s,jnp.zeros(6,dtype=jnp.int32),p)
            self.assertEqual(int(s.timestep),i+1)
            self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(s)))

    def test_actions(self):
        r=np.random.default_rng(7)
        for _ in range(100):
            a=sample_action(np.zeros(16),r)
            self.assertTrue(np.all((a>=0)&(a<np.array([3,3,3,3,2,2]))))
        z=np.zeros(16);z[2]=1000
        self.assertEqual(sample_action(z,r)[0],2)
    def test_bad_logits(self):
        for z in (np.zeros(15),np.full(16,np.nan)):
            with self.assertRaises(ValueError):sample_action(z,np.random.default_rng(0))
    def test_reject_code(self):
        p={k:np.zeros(s) for k,s in SHAPES.items()}
        for body in ('embed.b[0] = __import__("os")','embed.b[0] = [1]*5','unknown.W[0,0]=1'):
            with self.assertRaises(ValueError):reconstruct_native(p,'```edits\n'+body+'\n```')
    @unittest.skipUnless(REF.exists(),'released reference not installed')
    def test_released_reconstruction(self):
        s=ReplayStudy(REF)
        self.assertEqual(s.reconstruction['parameters'],171664)
        self.assertEqual(s.reconstruction['signed_zero_bit_differences'],3145)
        for p in s.policies.values():
            self.assertTrue(np.isfinite(p.logits(np.zeros(679))).all())
            self.assertTrue(all(not a.flags.writeable for a in p.weights.values()))
    @unittest.skipUnless(REF.exists(),'released reference not installed')
    def test_reject_incomplete_prompt(self):
        text=(REF/'runs/oneshot_repair/reply/prompt.txt').read_text(encoding='utf-8')
        with self.assertRaises(ValueError):parse_parent(text.replace('actor_head.b (16)','actor_head.b (15)'))

    def test_runner_with_one_tick_fixture(self):
        sys.path.insert(0,str(ROOT/'experiments'))
        import run_kinetix_replay as runner
        from skillforge.kinetix_replay import sha
        class Env:
            def reset(self,*args,**kwargs):return np.zeros(679),SimpleNamespace(last_distance=-1.)
            def step(self,*args):return np.zeros(679),SimpleNamespace(last_distance=-1.),1.,True,{'GoalR':True,'distance':0.}
        policy=SimpleNamespace(weights={'fixture':np.zeros(1)},tensor_sha256='unit-fixture',logits=lambda obs:np.zeros(16))
        study=SimpleNamespace(policies={'parent':policy,'edited':policy},reconstruction={})
        fakejax=SimpleNamespace(numpy=np,__version__='0.4.38',devices=lambda:[SimpleNamespace(platform='cpu')],
            random=SimpleNamespace(PRNGKey=lambda x:x),tree=SimpleNamespace(leaves=lambda x:[x.last_distance]))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'approved_manifest.json').write_text('{}')
            (out/'source_receipt.json').write_text(json.dumps({'status':'verified','manifest_sha256':sha(out/'approved_manifest.json')}))
            cfg=dict(resources={},episodes_per_arm=1024,level='fixture',action_seed_base=9100000,wall_guard_seconds=120)
            with patch.dict(sys.modules,{'jax':fakejax,'jax.numpy':np}),patch.dict('os.environ',{'JAX_PLATFORMS':'cpu','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}),patch.object(runner,'make_environment',return_value=(Env(),None,None)):
                runner.run(lambda:study,out,cfg)
            result=json.loads((out/'results.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['summaries']['parent']['outcomes']['success'],1024)
            rows=[json.loads(r) for r in (out/'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2048)
            self.assertEqual(rows[0]['seed'],rows[1]['seed'])
            self.assertEqual(len(list(out.glob('*trajectory.json'))),6)

if __name__=='__main__':unittest.main()
