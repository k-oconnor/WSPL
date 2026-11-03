"""Synthetic replay checks and a three-tick environment wiring test, not scoring."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.csail_breakout import (
    BreakoutPolicy,TENSORS,encode_observation,episode,file_hash,probabilities,sample_action,summarize,
)
from run_csail_breakout import run
try:import numpy as np
except ImportError:np=None


@unittest.skipIf(np is None,'optional NumPy unavailable')
class BreakoutReplayTests(unittest.TestCase):
    def artifact(self,root):
        path=Path(root)/'synthetic.npz'
        w={key:np.zeros(shape,dtype=np.float32) for key,shape in TENSORS.items()}
        w['actor/dense1/kernel'][0,0]=1
        w['actor/dense2/kernel'][0,0]=1
        w['actor/logits/kernel'][0,3]=2
        np.savez(path,**w)
        return path,w

    def test_layout_conversion_names_every_channel_and_uses_row_major_order(self):
        s=np.zeros((10,10,4),dtype=bool)
        for ch in range(4):s[ch,ch+1,ch]=True
        x=encode_observation(s).reshape(10,10,6)
        self.assertEqual(x.dtype,np.float32)
        self.assertEqual(float(x.sum()),4)
        for source,destination in ((0,0),(1,3),(2,4),(3,2)):
            self.assertEqual(x[source,source+1,destination],1)
            self.assertEqual(encode_observation(s)[(source*10+source+1)*6+destination],1)
        self.assertFalse(x[:,:,[1,5]].any())
        with self.assertRaises(ValueError):encode_observation(s.astype(float))

    def test_known_tanh_forward_and_frozen_tensor_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=self.artifact(tmp);p=BreakoutPolicy(path,file_hash(path))
            before=p.tensor_identity();x=np.zeros(600,dtype=np.float32);x[0]=1
            actual=p.logits(x)
            self.assertAlmostEqual(float(actual[3]),float(2*np.tanh(np.tanh(np.float32(1)))),places=6)
            self.assertEqual(np.count_nonzero(actual),1)
            self.assertEqual(before,p.tensor_identity())
            with self.assertRaises(ValueError):p.weights['actor/dense1/kernel'][0,0]=3
            p.uniform_control()
            self.assertTrue(np.array_equal(p.logits(x),np.zeros(6)))
            self.assertNotEqual(before,p.tensor_identity())

    def test_hash_shape_dtype_and_nan_fail_before_policy_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,w=self.artifact(tmp)
            with self.assertRaises(ValueError):BreakoutPolicy(path,'0'*64)
            for kind in ('shape','dtype','nan','missing'):
                altered={k:v.copy() for k,v in w.items()}
                key='actor/dense1/kernel'
                if kind=='shape':altered[key]=np.zeros((32,600),dtype=np.float32)
                elif kind=='dtype':altered[key]=altered[key].astype(np.float64)
                elif kind=='nan':altered[key][0,0]=np.nan
                else:altered.pop(key)
                changed=Path(tmp)/f'{kind}.npz';np.savez(changed,**altered)
                with self.assertRaises(ValueError):BreakoutPolicy(changed,file_hash(changed))

    def test_stable_softmax_and_sampling_retain_all_six_actions(self):
        p=probabilities(np.zeros(6))
        self.assertTrue(np.allclose(p,1/6))
        self.assertEqual(sample_action(p,0.),0)
        self.assertEqual(sample_action(p,.999),5)
        self.assertEqual(sample_action(probabilities([10000,-10000,0,0,0,0]),.5),0)
        with self.assertRaises(ValueError):sample_action(p,1.)
        with self.assertRaises(ValueError):probabilities([np.nan]*6)

    def fake_env(self,terminal_at=3):
        class FakeEnv:
            def seed(self,seed):self.seed_value=seed
            def reset(self):self.calls=[]
            def state(self):return np.zeros((10,10,4),dtype=bool)
            def act(self,action):
                self.calls.append(action)
                return 1.,len(self.calls)==terminal_at
        return FakeEnv()

    def test_sampling_seed_replay_and_terminal_vs_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=self.artifact(tmp);p=BreakoutPolicy(path,file_hash(path)).uniform_control()
            first=episode(p,self.fake_env(),env_seed=1,action_seed=2,max_steps=10)
            self.assertEqual(first,episode(p,self.fake_env(),env_seed=1,action_seed=2,max_steps=10))
            self.assertEqual((first['reward'],first['steps'],first['terminal'],first['truncated']),(3.,3,True,False))
            capped=episode(p,self.fake_env(terminal_at=99),env_seed=1,action_seed=2,max_steps=2)
            self.assertEqual((capped['steps'],capped['terminal'],capped['truncated']),(2,False,True))
            self.assertEqual(sum(capped['action_counts']),2)

    def test_clean_stop_never_promotes_partial_episode_to_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,_=self.artifact(tmp);p=BreakoutPolicy(path,file_hash(path))
            partial=episode(p,self.fake_env(),env_seed=1,action_seed=2,max_steps=10,should_stop=lambda:True)
            self.assertEqual(partial['status'],'incomplete')
            with self.assertRaises(ValueError):summarize([partial])
            with self.assertRaises(ValueError):summarize([])

    def test_summary_keeps_truncated_complete_trials_and_sample_se(self):
        rows=[dict(status='complete',reward=r,steps=2,truncated=True) for r in (0.,2.)]
        result=summarize(rows)
        self.assertEqual(result['mean'],1.)
        self.assertEqual(result['se'],1.)
        self.assertEqual(result['truncations'],2)

    @unittest.skipUnless(importlib.util.find_spec('minatar'),'isolated MinAtar dependency unavailable')
    def test_native_environment_three_tick_wiring_fixture(self):
        from minatar import Environment
        with tempfile.TemporaryDirectory() as tmp:
            path,_=self.artifact(tmp);p=BreakoutPolicy(path,file_hash(path)).uniform_control()
            result=episode(p,Environment('breakout',sticky_action_prob=.1),env_seed=41,action_seed=42,max_steps=3)
            self.assertEqual(result['steps'],3)
            self.assertEqual(sum(result['action_counts']),3)
            self.assertEqual(result['status'],'complete')

    def test_scientific_runner_requires_source_gate_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):run(lambda:None,Path(tmp),{})

    def runner_fixture(self,root):
        root=Path(root);output=root/'output';output.mkdir()
        path,_=self.artifact(root)
        (output/'approved_manifest.json').write_text('{}')
        (output/'source_receipt.json').write_text(json.dumps(dict(status='verified',manifest_sha256=file_hash(output/'approved_manifest.json'))))
        bundle=root/'bundle';bundle.mkdir()
        published=bundle/'exp5_write_from_scratch/eval/minatar_eval.json';published.parent.mkdir(parents=True)
        published.write_text(json.dumps({'breakout':{'hand':dict(mean=1.,se=1.,episodes=1000)}}))
        (bundle/'bundle.json').write_text(json.dumps(dict(files=[dict(path=published.relative_to(bundle).as_posix(),sha256=file_hash(published))])))
        config=dict(bundle=str(bundle),bundle_sha256=file_hash(bundle/'bundle.json'),dependencies_sha256='fixture',
                    auxiliary_files={},episodes_per_arm=1000,max_steps=10000,env_seed_base=1,action_seed_base=2000,wall_guard_seconds=30)
        calls=[]
        def construct():
            calls.append(1);return BreakoutPolicy(path,file_hash(path))
        def environment(*args,**kwargs):
            env=self.fake_env(1);env.env=SimpleNamespace(channels={'paddle':0,'ball':1,'trail':2,'brick':3});return env
        return output,config,construct,calls,environment

    def test_runner_mock_environment_records_both_arms_and_completed_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            output,config,construct,calls,environment=self.runner_fixture(tmp)
            with patch.dict('os.environ',{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}), \
                 patch.dict(sys.modules,{'minatar':SimpleNamespace(Environment=environment)}), \
                 patch('run_csail_breakout.dependency_identity',return_value={'sha256':'fixture'}):
                run(construct,output,config)
            result=json.loads((output/'results.json').read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(calls),2)
            self.assertEqual([a['summary']['episodes'] for a in result['arms'].values()],[1000,1000])
            self.assertNotEqual(result['arms']['released']['tensor_sha256'],result['arms']['uniform']['tensor_sha256'])
            self.assertEqual(len((output/'episodes.jsonl').read_text().splitlines()),2000)

    def test_runner_clean_stop_is_paused_not_a_complete_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            output,config,construct,_,environment=self.runner_fixture(tmp)
            (output/'control.json').write_text('{"stop":true}')
            with patch.dict('os.environ',{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}), \
                 patch.dict(sys.modules,{'minatar':SimpleNamespace(Environment=environment)}), \
                 patch('run_csail_breakout.dependency_identity',return_value={'sha256':'fixture'}):
                run(construct,output,config)
            result=json.loads((output/'results.json').read_text())
            self.assertEqual(result['status'],'paused')
            self.assertEqual(result['stop_reason'],'user_stop')
            self.assertIsNone(result['comparison'])
            self.assertFalse(result['arms']['released']['complete'])

    def test_dependency_mismatch_blocks_before_any_constructor_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            output,config,construct,calls,environment=self.runner_fixture(tmp)
            with patch.dict('os.environ',{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}), \
                 patch.dict(sys.modules,{'minatar':SimpleNamespace(Environment=environment)}), \
                 patch('run_csail_breakout.dependency_identity',return_value={'sha256':'different'}):
                with self.assertRaisesRegex(ValueError,'dependency identity'):run(construct,output,config)
            self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
