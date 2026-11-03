"""Fake collector/environment integration; no model or gameplay calls."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import run_astra_breakout as runner
from skillforge.csail_breakout import file_hash
from skillforge.native_weight_edits import NativeEditedPolicy


class RunnerTests(unittest.TestCase):
    def execute_fake(self,collect):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/'out';out.mkdir()
            (out/'approved_manifest.json').write_text('{}')
            (out/'source_receipt.json').write_text(json.dumps(dict(status='verified',manifest_sha256=file_hash(out/'approved_manifest.json'))))
            cli=p/'fake_cli';cli.write_text('fake')
            bindings=p/'bindings.json';bindings.write_text(json.dumps(dict(codex_cli=str(cli))))
            prompt=p/'prompt';prompt.write_text('task')
            config=dict(bound_files={},dependencies_sha256='dep',draws=3,episodes=1000,max_steps=10000,
                prompt=str(prompt),local_paths=str(bindings),cli_sha256=file_hash(cli),
                catalog='unused',instructions='unused',call_guard_seconds=1200,wall_guard_seconds=7200)
            constructor_calls=[];seeds=[]
            def construct():constructor_calls.append(1);return NativeEditedPolicy()
            def episode(*args,**kw):
                seeds.append((kw['env_seed'],kw['action_seed']))
                return dict(status='complete',reward=1,steps=1,truncated=False)
            fake=SimpleNamespace(Environment=lambda *a,**kw:SimpleNamespace(env=SimpleNamespace(channels={'paddle':0,'ball':1,'trail':2,'brick':3})))
            with patch.dict(os.environ,{k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),patch.dict(sys.modules,{'minatar':fake}),patch.object(runner,'dependency_identity',return_value={'sha256':'dep'}),patch.object(runner,'collect',side_effect=collect),patch.object(runner,'episode',side_effect=episode):
                runner.run(construct,out,config)
            return json.loads((out/'results.json').read_text()),len(constructor_calls),seeds

    def test_three_independent_draws_paired_evaluation_seeds(self):
        ids=iter(range(3))
        def collect(*args):
            Path(args[3]).mkdir()
            return '```edits\nb3[:]=0\n```',dict(thread_id=str(next(ids)),usage={'output_tokens':10})
        result,calls,seeds=self.execute_fake(collect)
        self.assertEqual(result['status'],'complete');self.assertEqual(calls,3)
        self.assertEqual(result['completed_evaluations'],3)
        self.assertEqual(seeds[:1000],seeds[1000:2000]);self.assertEqual(seeds[:1000],seeds[2000:])

    def test_failure_stops_without_retry_and_keeps_denominator(self):
        def failure(*args):raise RuntimeError('quota failure')
        result,calls,seeds=self.execute_fake(failure)
        self.assertEqual(result['status'],'incomplete');self.assertEqual(calls,1)
        self.assertEqual([t['status'] for t in result['trials']],['collection_failed','not_attempted','not_attempted'])
        self.assertEqual(result['planned_draws'],3);self.assertEqual(seeds,[])

    def test_invalid_answers_preserved_not_retried(self):
        ids=iter(range(3))
        def invalid(*args):return 'no weights',dict(thread_id=str(next(ids)))
        result,calls,seeds=self.execute_fake(invalid)
        self.assertEqual(calls,3);self.assertEqual(result['completed_evaluations'],0)
        self.assertTrue(all(t['status']=='invalid_proposal' for t in result['trials']))
        self.assertEqual(seeds,[])
