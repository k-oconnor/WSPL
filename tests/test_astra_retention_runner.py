import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import run_astra_retention as runner
from test_closed_loop_runner import Model,Tokenizer
from skillforge.authoring_sft import adapter_state,frozen_hash
from skillforge.controller_author import make_spec,construct,teacher,KINDS
from skillforge.adaptation_pilot import tasks
from skillforge.retention_pilot import DEFAULT_SOURCE


class RunnerTests(unittest.TestCase):
    def test_full_both_regimes_and_return(self):
        specs={k:make_spec(k,20) for k in KINDS};specs['move'].update(gain=.73,speed=.07)
        blocks={k:construct(teacher(v),v) for k,v in specs.items()}
        repair=construct(teacher(specs['move']).replace('0.73 * x','0.6 * y').replace('0.73 * y','-0.9 * x'),specs['move'])
        tokenizer=Tokenizer();model=Model(tokenizer,specs['move'])
        source=DEFAULT_SOURCE.replace('original = 1','original = dx').replace('repaired = 0','repaired = dy')
        commands=[dict(tool='install',prediction='both regimes',source=source,command=[.005,0.]),
                  dict(tool='evaluate',prediction='retain and improve'),dict(tool='finish',prediction='freeze')]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);ck=p/'executive.pt';rp=p/'repair.json';repair.save(rp)
            torch.save(dict(adapter=adapter_state(model),base_hash=frozen_hash(model),manifest_sha256='synthetic'),ck)
            cfg=dict(resources={},library={},mapping=[[0.,-.8],[1.2,0.]],repair=str(rp),executives={'113':str(ck)},
                     executive_manifest='synthetic',wall_guard_seconds=120,cli='',catalog='',instructions='')
            calls=[]
            def collect(*args):calls.append(args[4]);return json.dumps(commands[len(calls)-1]),dict(usage={})
            with patch.object(runner,'load_library',return_value=(blocks,specs)),patch.object(runner,'collect',side_effect=collect),patch.object(runner,'tasks',side_effect=lambda start,counts:tasks(start,[2])):
                runner.run(lambda:SimpleNamespace(model=model,tokenizer=tokenizer),p,cfg)
            r=json.loads((p/'results.json').read_text());rows=json.loads((p/'scores.json').read_text())
            self.assertTrue(r['feasibility_passed']);self.assertEqual(len(rows),12)
            self.assertTrue(all(x['task_success'] for x in rows if x['arm']=='selected'))
            self.assertNotIn('mapping',calls[0]);self.assertNotIn('reference_source',calls[0])
            self.assertEqual(r['paired']['113/original_return']['regressions'],0)


if __name__=='__main__':unittest.main()
