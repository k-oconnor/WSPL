"""Source-gated supervisor pilot. Generated research code lives outside source pin."""
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import time
from skillforge.supervisor_workspace import Budget,freeze,save,digest
from skillforge.supervisor_session import SupervisorSession
from skillforge.supervisor_transport import ToolBridge,linux_path
from skillforge.reliable_json import status


def run(constructor,output_dir,config):
    out=Path(output_dir).resolve();started=time.time()
    def report(phase,**kw):status(out/'status.json',dict(status=phase,elapsed_seconds=time.time()-started,**kw))
    try:
        report('verifying')
        for index,(path,h) in enumerate(config['resources'].items()):
            if digest(path)!=h:raise ValueError('Bound resource changed: '+path)
            if index%250==0:report('verifying',checked=index+1,total=len(config['resources']))
        if 'linux_runtime_manifest' in config:
            report('verifying_linux_runtime')
            checked=subprocess.run(['wsl.exe','-d','Ubuntu','--',config['runtime']+'/bin/python',
                linux_path(config['runtime_verifier']),linux_path(config['linux_runtime_manifest']),
                config['linux_runtime_sha256']],capture_output=True,text=True,encoding='utf8',timeout=900)
            (out/'linux_verification.stdout.log').write_text(checked.stdout,encoding='utf8')
            (out/'linux_verification.stderr.log').write_text(checked.stderr,encoding='utf8')
            if checked.returncode:raise ValueError('Native Linux runtime verification failed; no model calls')
            save(out/'linux_runtime_receipt.json',json.loads(checked.stdout))
        study=constructor();assert study.kind=='astra_supervisor_v1'
        work=out/'development';work.mkdir()
        baseline=out/'baseline_candidate';baseline.mkdir()
        shutil.copyfile(config['template'],baseline/'policy.py')
        save(baseline/'submission.json',dict(entrypoint='policy.py',description='Saved ordinary-practice Qwen executive'))
        baseline_bundle=freeze(baseline,out/'baseline_store',dict(kind='ordinary_practice'))
        shutil.copytree(baseline,work/'candidate')
        # Plain files tell Astra where explicitly mounted dependencies are located.
        save(work/'resources.json',dict(mounts=list(config['readonly']),model='/inputs/model',
            starting_adapter='/inputs/parent',motor_artifact='/inputs/motor',source='/inputs/code',
            original_adapter='/inputs/original',taught_adapter='/inputs/taught'))
        (work/'START_HERE.md').write_text(config['workbench_help'],encoding='utf8')
        budget=Budget(out/'budget.json',**config['limits'])
        bridge=ToolBridge(config,work,out/'budget.json',config['limits'])
        brief=Path(config['brief']).read_text(encoding='utf8')+'\n'+config['workbench_help']
        session=SupervisorSession(out/'supervisor',work,budget,brief,bridge.execute,bridge.collect)
        while session.state['phase']=='ready':
            report('supervising',turn=session.state['turn'],calls_used=budget.state['calls_used'])
            session.step()
        if session.state['phase']!='submitted':raise RuntimeError('No final submission')
        submission=Path(session.state['submission'])
        # No final seeds exist until authoring has ended. Do not mount this folder.
        sealed=out/'sealed';sealed.mkdir()
        seed=secrets.randbelow(1_000_000_000)+100_000_000
        missions=[dict(seed=seed+i,count=3+i%3,condition=c)
                  for i,c in enumerate([c for c in ('ordinary','changed','switch','return') for _ in range(8)])]
        save(sealed/'missions.json',missions)
        save(out/'frozen.json',dict(submission=submission.name,baseline=baseline_bundle.name,
            sealed_sha256=digest(sealed/'missions.json'),supervisor_responses=session.state['turn']))
        evaluation=dict(runtime=config['runtime'],python_home=config['python_home'],
            deployment_inputs={k:linux_path(config['readonly'][k]) for k in ('model','parent','code')},
            deployment_worker=linux_path(config['deployment_worker']),motor_artifact=linux_path(config['readonly']['motor']),
            evaluation_seconds=7200,final_guard_seconds=14400,missions=linux_path(sealed/'missions.json'),
            output=linux_path(out/'evaluation'),bundles={'ordinary_practice':linux_path(baseline_bundle),'submission':linux_path(submission)})
        save(sealed/'evaluation.json',evaluation);report('evaluating',supervisor_removed=True)
        with (out/'evaluation.stdout.log').open('wb') as stdout,(out/'evaluation.stderr.log').open('wb') as stderr:
            completed=subprocess.Popen(['wsl.exe','-d','Ubuntu','--',config['runtime']+'/bin/python',
                linux_path(config['final_worker']),linux_path(sealed/'evaluation.json')],stdout=stdout,stderr=stderr)
            while completed.poll() is None:
                progress={}
                p=out/'evaluation/progress.json'
                try:progress=json.loads(p.read_text())
                except (OSError,ValueError):pass
                report('evaluating',supervisor_removed=True,progress=progress)
                time.sleep(2)
        if completed.returncode:raise RuntimeError('Final evaluator stopped; no automatic retry')
        result=json.loads((out/'evaluation/results.json').read_text())
        save(out/'results.json',result);report('complete',results=result['summary'],exploratory_gate=result['exploratory_gate'])
    except BaseException as error:
        report('failed',error=repr(error));raise
