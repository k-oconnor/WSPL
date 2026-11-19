import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import numpy as np
import run_executive_teaching as runner
from skillforge.executive_teaching import MonitoredBackend


class PipelineTests(unittest.TestCase):
    def test_private_physical_monitor(self):
        class Backend:
            _eef_site=0
            env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(site_xpos=np.array([[0.,0.,0.]]))))
            def sensors(self):return dict(timestamp_s=1.)
            def step(self,c):pass
        m=MonitoredBackend(Backend());m.goals={'first':[0,0,0],'second':[1,0,0]};m.target='first'
        for _ in range(3):m.step([0,0,0,-1])
        self.assertEqual(m.visits,['first']);self.assertEqual(m.order_violations,[])
        self.assertEqual(len(m.true_trace),3)
    def _mock_pipeline(self,ceiling=False):
        events=[]
        class FakeSim:
            def __init__(self,*args):pass
            def call(self,r):
                if r['op']=='start':
                    self.label=Path(r['output']).name
                    if Path(r['output']).name.startswith(('unchanged_','rehearsal_','taught_')):
                        self_outer.assertEqual(events.count('fit'),2)
                        self_outer.assertTrue((Path(r['output']).parent/'frozen.json').exists())
                    self.public=dict(observation=dict(selected_module=None,last=None,requested_order=['g'],observed_visits=[]),documentation={'m':'Original motor block'},finished=False)
                    return self.public
                if r['op']=='act':return dict(self.public,finished=True,outcome=dict(status='stopped'))
                if r['op']=='finish':return dict(success=ceiling or not self.label.startswith('acquisition_'),probes=0,physical_ticks=0,decisions=1)
                raise AssertionError(r)
            def close(self):events.append('closed')
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return 'synthetic'
            def __call__(self,*args,**kwargs):return SimpleNamespace(input_ids=torch.tensor([[1,2]]))
            def decode(self,*args,**kwargs):return '{"op":"stop"}'
            eos_token_id=0
        class Model:
            def eval(self):pass
            def gradient_checkpointing_disable(self):pass
            def generate(self,ids,**kwargs):return torch.tensor([[1,2,3]])
        parent=dict(adapter={'synthetic':torch.tensor([1.])},base_hash='base',manifest_sha256='parent')
        def fake_fit(model,tokenizer,parent,ordinary,teacher,arm,out,*args):
            events.append('fit');out.mkdir();(out/'adapter_final.pt').write_bytes(arm.encode())
        self_outer=self
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);(out/'source_receipt.json').write_text('{"manifest_sha256":"synthetic"}');(out/'approved_manifest.json').write_text('{}')
            p=out/'parent.pt';p.write_bytes(b'p');a=out/'motor.json';a.write_bytes(b'a')
            config=dict(resources={str(p):hashlib.sha256(b'p').hexdigest(),str(a):hashlib.sha256(b'a').hexdigest()},parent=str(p),parent_manifest='parent',motor_artifact=str(a),wall_guard_seconds=60,cli='none',catalog='none',instructions='none')
            def collect(*args,**kwargs):events.append('teacher');return '{"prediction":"finish","action":{"op":"stop"}}',dict(usage={})
            original_to=torch.Tensor.to
            def cpu_to(value,*args,**kwargs):
                if args and args[0]=='cuda':return value
                return original_to(value,*args,**kwargs)
            with patch.object(runner,'Simulator',FakeSim),patch.object(runner,'collect',collect),patch.object(runner,'fit',fake_fit),patch.object(runner,'encode'),patch.object(runner,'frozen_hash',return_value='base'),patch.object(runner,'restore_adapter'),patch.object(runner,'adapter_state',return_value=parent['adapter']),patch.object(runner.torch,'load',return_value=parent),patch.object(torch.Tensor,'to',cpu_to):
                runner.run(lambda:SimpleNamespace(model=Model(),tokenizer=Tokenizer()),out,config)
            result=json.loads((out/'results.json').read_text())
            if ceiling:
                self.assertEqual(result['phase'],'no_acquisition_adaptation_deficit')
                self.assertEqual(events.count('teacher'),0);self.assertEqual(events.count('fit'),0)
                self.assertFalse((out/'scores.json').exists());return
            self.assertEqual(result['status'],'complete');self.assertFalse(result['learning_signal_passed'])
            self.assertEqual(events.count('teacher'),8);self.assertEqual(events.count('fit'),2)
            self.assertEqual(len(json.loads((out/'scores.json').read_text())),96)
            self.assertEqual(events[-1],'closed')

    def test_mock_full_pipeline_freezes_before_evaluation(self):self._mock_pipeline()
    def test_no_teacher_spend_without_acquisition_deficit(self):self._mock_pipeline(ceiling=True)


if __name__=='__main__':unittest.main()
