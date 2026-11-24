"""Continuation and collector checks with fake authors/evaluation only."""
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
from skillforge.robot_collector import settings,check_event
from skillforge.fresh_robot import sha
from skillforge.teaching_training import state_hash
from skillforge.wspl_lora import describe
import run_robot_authoring_v2 as runner


class CollectorRepairTests(unittest.TestCase):
    def test_required_host_enabled_even_when_tools_disabled(self):
        values=settings('catalog','instructions','supervisor')
        self.assertIn('features.code_mode_host=true',values)
        self.assertNotIn('features.code_mode_host=false',values)
        self.assertIn('features.incremental_tools=false',values)
        self.assertIn('features.incremental_tools_legacy=false',values)
    def test_runtime_error_retains_actual_cause(self):
        with self.assertRaisesRegex(ValueError,'required host is disabled'):
            check_event(dict(type='item.completed',item=dict(type='error',message='required host is disabled')))


class RecoveryOrchestrationTests(unittest.TestCase):
    def test_reuses_prefix_without_training_or_new_acquisition(self):
        bank={'p.lora_A':np.ones((2,3),np.float32),'p.lora_B':np.zeros((4,2),np.float32)}
        state={k:torch.from_numpy(v) for k,v in bank.items()}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'source_receipt.json').write_text(json.dumps(dict(manifest_sha256='fixture')))
            cfg=dict(resources={},acquisition_seed=73300000,development_seed=73400000,final_seed=73500000,
                development_wall_seconds=100,final_wall_seconds=100,compute_seconds=100,
                prior_compute_seconds=12.,prior_elapsed_seconds=13.,parent_checkpoint='unused',
                base_hash='base',parent_factor_hash=state_hash(state))
            events=[]
            def recover(config,out,guard):
                common=out/'common_inputs';common.mkdir();np.savez(common/'parent.npz',**bank)
                gradient=out/'reused_gradient.npz';np.savez(gradient,**bank)
                base=[dict(**m,success=True,model_calls=1,invalid_tokens=0,start_sha256=str(m['seed'])) for m in runner.panel(73400000,4)]
                (common/'parent_development.json').write_text(json.dumps(runner.summary(base)))
                (out/'common_input_hashes.json').write_text(json.dumps({p.name:sha(p) for p in common.iterdir()}))
                catalog=describe(bank);catalog.update(diagnostic_projection='fixture',feature_shapes={})
                return common,[{}]*255,catalog,base,gradient
            def session(*a,**k):events.append('author');return None
            def evaluate(model,tokenizer,config,folder,manifest,pin,missions,guard,report):
                self.assertTrue((out/'all_frozen.json').exists());self.assertEqual(events.count('author'),6)
                self.assertEqual(len(missions),64);events.append('final')
                return [dict(**m,success=True,model_calls=1,invalid_tokens=0,start_sha256=str(m['seed'])) for m in missions]
            replacements=dict(load_prefix=recover,run_session=session,evaluate=evaluate,
                restore_adapter=lambda *a:None,adapter_state=lambda m:state,frozen_hash=lambda m:'base')
            with patch.multiple(runner,**replacements),patch.object(runner.torch,'load',return_value={'adapter':state}),\
                    patch.object(runner,'fit',side_effect=AssertionError('Control must not retrain')),\
                    patch.object(runner,'acquire',side_effect=AssertionError('Acquisition must not rerun')),\
                    patch.object(runner,'features',side_effect=AssertionError('Diagnostics must not rerun')):
                runner.run(lambda:SimpleNamespace(model=object(),tokenizer=object()),out,cfg)
            result=json.loads((out/'results.json').read_text())
            self.assertEqual(result['new_gradient_updates'],0);self.assertEqual(result['reused_gradient_updates'],384)
            self.assertEqual(result['prior_attempted_author_invocations'],1);self.assertEqual(events.count('final'),2)
            self.assertGreaterEqual(result['development_compute_seconds'],12)


if __name__=='__main__':unittest.main()
