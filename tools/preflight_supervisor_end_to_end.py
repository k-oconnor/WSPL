"""Windows -> Linux scripted supervisor -> frozen policy -> physical fixture.

No Astra, pretrained Qwen or scientific benchmark run; policy deliberately stops.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.supervisor_session import SupervisorSession
from skillforge.supervisor_workspace import Budget,save
from skillforge.supervisor_transport import ToolBridge,linux_path


def main():
    runtime='/home/kevin/venvs/wspl-supervisor'
    home='/home/kevin/.local/share/uv/python/cpython-3.11.17-linux-x86_64-gnu'
    parent=ROOT/'data/supervisor_engineering';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        root=Path(tmp);work=root/'work';work.mkdir()
        limits=dict(wall_seconds=180,gpu_seconds=120,calls=2)
        b=Budget(root/'budget.json',**limits)
        config=dict(runtime=runtime,python_home=home,readonly={},job_worker=str(ROOT/'tools/supervisor_job_worker.py'))
        bridge=ToolBridge(config,work,root/'budget.json',limits)
        policy='''class Policy:
 def __init__(self,root,declaration): pass
 def reset(self,public): pass
 def act(self,observation): return {"op":"stop"}
 def close(self): pass
'''
        program="from pathlib import Path\nimport json\np=Path('/work/candidate');p.mkdir()\n(p/'policy.py').write_text("+repr(policy)+")\n(p/'submission.json').write_text(json.dumps({'entrypoint':'policy.py'}))\nprint('synthetic candidate created')"
        def collect(brief,history,folder):
            return json.dumps(dict(tool='submit',directory='candidate',description='synthetic stop policy')) if history else json.dumps(dict(tool='python',code=program,gpu=False,seconds=10))
        session=SupervisorSession(root/'session',work,b,'synthetic',bridge.execute,collect)
        session.step();session.step();assert session.state['phase']=='submitted'
        missions=root/'missions.json';save(missions,[dict(seed=998,count=2,condition='ordinary')])
        bundle=linux_path(session.state['submission'])
        evaluation=dict(runtime=runtime,python_home=home,deployment_inputs={},
            deployment_worker=linux_path(ROOT/'tools/supervisor_deployment_worker.py'),
            motor_artifact=linux_path(ROOT.parent/'WSPL-factory-v1/results/factory_pilot_v1/artifact_final.json'),
            evaluation_seconds=60,missions=linux_path(missions),output=linux_path(root/'evaluation'),
            bundles={'ordinary_practice':bundle,'submission':bundle})
        save(root/'evaluation.json',evaluation)
        result=subprocess.run(['wsl.exe','-d','Ubuntu','--',runtime+'/bin/python',linux_path(ROOT/'tools/supervisor_final_worker.py'),linux_path(root/'evaluation.json')],capture_output=True,text=True,timeout=120)
        assert result.returncode==0,result.stderr
        scores=json.loads((root/'evaluation/results.json').read_text())
        assert len(scores['rows'])==2 and all(not r['success'] for r in scores['rows'])
        report=dict(status='passed',scripted_supervisor=True,python_job=True,frozen_submission=True,
            offline_policy=True,paired_physics_starts=True,deliberate_failure_scored_correctly=True,
            model_calls=0,scientific_episodes=0)
        save(ROOT/'docs/wspl/supervisor_end_to_end_preflight.json',report);print(json.dumps(report))


if __name__=='__main__':main()
