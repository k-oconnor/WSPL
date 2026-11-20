import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_astra_supervisor as runner
from skillforge.supervisor_workspace import freeze


class RunnerTests(unittest.TestCase):
    def test_seal_only_after_submission_and_no_teacher_at_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir()
            template=root/'policy.py';template.write_text('class Policy: pass')
            brief=root/'brief';brief.write_text('synthetic')
            config=dict(resources={},template=str(template),brief=str(brief),workbench_help='synthetic',
                limits=dict(wall_seconds=60,gpu_seconds=30,calls=2),runtime='/runtime',python_home='/python',
                readonly={k:str(root/k) for k in ('model','parent','code','motor')},
                deployment_worker=str(root/'worker.py'),final_worker=str(root/'final.py'))
            events=[]
            class Session:
                def __init__(self,*args):self.state=dict(phase='ready',turn=0)
                def step(self):
                    self_outer.assertFalse((out/'sealed').exists())
                    bundle=freeze(out/'development/candidate',out/'store',{})
                    self.state=dict(phase='submitted',turn=1,submission=str(bundle));events.append('submitted')
            class Process:
                returncode=0
                def __init__(self,*args,**kwargs):
                    self_outer.assertEqual(events,['submitted'])
                    frozen=json.loads((out/'frozen.json').read_text())
                    self_outer.assertIn('sealed_sha256',frozen)
                    missions=json.loads((out/'sealed/missions.json').read_text())
                    self_outer.assertEqual(len(missions),32)
                    (out/'evaluation').mkdir()
                    (out/'evaluation/results.json').write_text('{"summary":{},"exploratory_gate":false}')
                    events.append('evaluation')
                def poll(self):return 0
            self_outer=self
            with patch.object(runner,'ToolBridge',return_value=SimpleNamespace(execute=None,collect=None)),patch.object(runner,'SupervisorSession',Session),patch.object(runner.subprocess,'Popen',Process):
                runner.run(lambda:SimpleNamespace(kind='astra_supervisor_v1'),out,config)
            self.assertEqual(events,['submitted','evaluation'])
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'complete')
    def test_changed_resource_prevents_constructor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);resource=root/'resource';resource.write_text('changed')
            with patch.object(runner,'ToolBridge') as bridge:
                with self.assertRaises(ValueError):runner.run(lambda:self.fail('constructed'),root,{'resources':{str(resource):'incorrect'}})
                bridge.assert_not_called()


if __name__=='__main__':unittest.main()
