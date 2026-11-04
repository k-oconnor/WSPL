"""Fake collector/environment queue checks; no model or game calls."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import run_breakout_matched as runner
from skillforge.csail_breakout import file_hash


class QueueTests(unittest.TestCase):
    def exercise(self,fail=False,invalid_wspl=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir()
            (out/'approved_manifest.json').write_text('{}')
            (out/'source_receipt.json').write_text(json.dumps(dict(status='verified',manifest_sha256=file_hash(out/'approved_manifest.json'))))
            cli=root/'fake';cli.write_text('fake')
            local=root/'paths';local.write_text(json.dumps(dict(codex_cli=str(cli))))
            prompts={}
            for arm in ('weights','code','wspl'):
                f=root/(arm+'.txt');f.write_text('common task\nselected '+arm);prompts[arm]=str(f)
            config=dict(bound_files={},dependencies_sha256='test',local_paths=str(local),cli_sha256=file_hash(cli),
                episodes=1000,max_steps=10000,call_guard_seconds=1200,wall_guard_seconds=14400,
                prompts=prompts,catalog='unused',instructions='unused')
            calls=[];seeds=[]
            class FakePolicy:
                def __init__(self):
                    calls.append(1);self.weights={'test':np.zeros(1,dtype=np.float32)};self.program=None
                    self.artifact=SimpleNamespace(save=lambda p:p.write_text('{}'))
                def load_answer(self,arm,answer):
                    if invalid_wspl and arm=='wspl':raise ValueError('invalid text')
                    return dict(tensor_sha256='same')
                def tensor_identity(self):return 'same'
            def collector(*args):
                if fail:raise RuntimeError('quota')
                Path(args[3]).mkdir();return 'answer',dict(thread_id=str(len(calls)),usage={'output_tokens':1})
            def episode(*args,**kwargs):
                seeds.append((kwargs['env_seed'],kwargs['action_seed']))
                return dict(status='complete',reward=200,steps=1,truncated=False)
            env=SimpleNamespace(Environment=lambda *a,**kw:SimpleNamespace(env=SimpleNamespace(channels={'paddle':0,'ball':1,'trail':2,'brick':3})))
            with patch.dict(os.environ,{k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),patch.dict(sys.modules,{'minatar':env}),patch.object(runner,'dependency_identity',return_value={'sha256':'test'}),patch.object(runner,'collect',side_effect=collector),patch.object(runner,'episode',side_effect=episode):
                runner.run(FakePolicy,out,config)
            return json.loads((out/'results.json').read_text()),len(calls),seeds

    def test_nine_calls_same_eval_conditions(self):
        result,calls,seeds=self.exercise()
        self.assertEqual(calls,9);self.assertEqual(result['status'],'complete')
        self.assertEqual(len(seeds),9000)
        for i in range(9):self.assertEqual(seeds[i*1000:(i+1)*1000],seeds[:1000])
        self.assertEqual(seeds[0],(710000,810000))
        self.assertEqual(result['comparisons']['internal_feasibility_decision'],'go')

    def test_fail_fast_no_hidden_retry(self):
        result,calls,seeds=self.exercise(fail=True)
        self.assertEqual(calls,1);self.assertEqual(result['status'],'incomplete');self.assertEqual(seeds,[])
        self.assertEqual(result['planned_invocations'],9);self.assertEqual(result['attempted_invocations'],1)
        self.assertEqual(sum(t['status']=='not_attempted' for t in result['trials']),8)

    def test_bad_source_preserved_queue_continues(self):
        result,calls,seeds=self.exercise(invalid_wspl=True)
        self.assertEqual(calls,9);self.assertEqual(result['status'],'complete')
        self.assertEqual(len(seeds),6000)
        self.assertEqual(result['comparisons']['arms']['wspl']['invalid'],3)
        self.assertEqual(result['comparisons']['arms']['wspl']['failure_adjusted_mean'],0)
