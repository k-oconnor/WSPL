"""Fake orchestration confirms neutral inputs and freeze-before-final boundary."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.fresh_robot import sha
from skillforge.teaching_training import state_hash
from skillforge.wspl_lora import describe
import run_robot_authoring_access_v1 as runner


class AccessRunnerTests(unittest.TestCase):
    def test_new_authors_no_retraining_neutral_inputs_fresh_final_after_freeze(self):
        bank={'p.lora_A':np.ones((2,3),np.float32),'p.lora_B':np.zeros((4,2),np.float32)}
        state={k:torch.from_numpy(v) for k,v in bank.items()}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'source_receipt.json').write_text(json.dumps(dict(manifest_sha256='fixture')))
            cfg=dict(resources={},acquisition_seed=73300000,development_seed=73400000,final_seed=73600000,
                development_wall_seconds=100,final_wall_seconds=100,compute_seconds=100,parent_checkpoint='unused',
                base_hash='base',parent_factor_hash=state_hash(state))
            events=[]
            def recover(config,out,guard):
                common=out/'common_inputs';common.mkdir();np.savez(common/'parent.npz',**bank);np.savez(common/'features.npz',X=np.zeros(1))
                gradient=out/'reused_gradient.npz';np.savez(gradient,**bank)
                base=[dict(**m,success=True,model_calls=1,invalid_tokens=0,start_sha256=str(m['seed'])) for m in runner.panel(73400000,4)]
                (common/'parent_development.json').write_text(json.dumps(runner.summary(base)))
                catalog=describe(bank);catalog.update(diagnostic_projection='p',feature_shapes={})
                (common/'catalog.json').write_text(json.dumps(catalog))
                (out/'common_input_hashes.json').write_text(json.dumps({p.name:sha(p) for p in common.iterdir()}))
                return common,[{}]*255,catalog,base,gradient
            def session(arm,common,packet,config,folder,*a,**kw):
                self.assertFalse((common/'features.npz').exists());self.assertNotIn('diagnostic_projection',json.loads((common/'catalog.json').read_text()))
                self.assertEqual(packet['projections'],['p']);self.assertIsNotNone(kw['query_candidate'])
                folder.mkdir();(folder/'frozen.json').write_text(json.dumps(dict(inspection_attempts=0)))
                events.append('author');return None
            def evaluate(model,tokenizer,config,folder,manifest,pin,missions,guard,report):
                self.assertTrue((out/'all_frozen.json').exists());self.assertEqual(events.count('author'),6)
                self.assertTrue(all(m['seed']>=73600000 for m in missions));events.append('final')
                return [dict(**m,success=True,model_calls=1,invalid_tokens=0,start_sha256=str(m['seed'])) for m in missions]
            patches=dict(load_prefix=recover,run_session=session,evaluate=evaluate,restore_adapter=lambda *a:None,
                adapter_state=lambda m:state,frozen_hash=lambda m:'base')
            with patch.multiple(runner,**patches),patch.object(runner.torch,'load',return_value={'adapter':state}),\
                 patch.object(runner,'fit',side_effect=AssertionError('No training')),\
                 patch.object(runner,'acquire',side_effect=AssertionError('No collection')),\
                 patch.object(runner,'features',side_effect=AssertionError('No old diagnostics')):
                runner.run(lambda:SimpleNamespace(model=object(),tokenizer=object()),out,cfg)
            result=json.loads((out/'results.json').read_text())
            self.assertEqual(result['new_gradient_updates'],0);self.assertEqual(result['reused_gradient_updates'],384)
            self.assertEqual(result['study'],'broader_access');self.assertEqual(events.count('final'),2)
    def test_old_final_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Fresh reserved'):
            runner.validate_splits(dict(acquisition_seed=73300000,development_seed=73400000,final_seed=73500000))


if __name__=='__main__':unittest.main()
