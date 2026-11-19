import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import run_astra_adaptation as runner
from test_closed_loop_runner import Model,Tokenizer
from skillforge.authoring_sft import adapter_state,frozen_hash
from skillforge.controller_author import make_spec,construct,teacher,KINDS
from skillforge.adaptation_pilot import tasks


class RunnerTests(unittest.TestCase):
    def test_bounded_session_freeze_and_deployment(self):
        specs={k:make_spec(k,20) for k in KINDS};specs['move'].update(gain=.73,speed=.07)
        blocks={k:construct(teacher(v),v) for k,v in specs.items()}
        source=teacher(specs['move']).replace('0.73 * x','0.6 * y').replace('0.73 * y','-0.9 * x')
        tokenizer=Tokenizer();model=Model(tokenizer,specs['move'])
        commands=[dict(tool='probe',prediction='measure both axes',commands=[[.01,0],[0,.01]]),
                  dict(tool='author',prediction='ask saved author',request='synthetic zero block'),
                  dict(tool='fit',prediction='synthetic zero target',updates=1,rows=[dict(inputs=dict(x=0.,y=0.),outputs=dict(vx=0.,vy=0.,grip=0.))]),
                  dict(tool='install_wspl',prediction='repair',source=source),
                  dict(tool='evaluate',prediction='recovery'),dict(tool='finish',prediction='freeze')]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);ck=p/'executive.pt'
            torch.save(dict(adapter=adapter_state(model),base_hash=frozen_hash(model),manifest_sha256='synthetic'),ck)
            cfg=dict(resources={},library={},mapping=[[0.,-.8],[1.2,0.]],reference_source=source,
                     executives={'113':str(ck)},executive_manifest='synthetic',wall_guard_seconds=120,
                     cli='',catalog='',instructions='',author_adapter=str(ck),author_manifest='synthetic')
            calls=[]
            def collect(*args):
                calls.append(args[4]);return json.dumps(commands[len(calls)-1]),dict(usage={})
            def small_tasks(start,counts):return tasks(start,[2])
            with patch.object(runner,'load_library',return_value=(blocks,specs)),patch.object(runner,'collect',side_effect=collect),patch.object(runner,'tasks',side_effect=small_tasks):
                runner.run(lambda:SimpleNamespace(model=model,tokenizer=tokenizer),p,cfg)
            result=json.loads((p/'results.json').read_text());scores=json.loads((p/'scores.json').read_text())
            self.assertEqual(result['status'],'complete');self.assertEqual(result['author_turns'],6)
            self.assertEqual(result['remaining']['fit_updates'],4095)
            self.assertEqual(result['remaining']['author_calls'],3)
            self.assertEqual(len(scores),6);self.assertTrue((p/'frozen.json').exists())
            self.assertFalse(next(r for r in scores if r['arm']=='unchanged' and r['panel']=='changed')['task_success'])
            self.assertTrue(next(r for r in scores if r['arm']=='repaired' and r['panel']=='changed')['task_success'])
            self.assertNotIn('reference_source',calls[0])
            self.assertNotIn('movement_final',calls[-1])


if __name__=='__main__':unittest.main()
