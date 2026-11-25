"""Hidden watchdog; requires a separately registered exact source approval."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from source_consistency import REGISTRY,contained,verify


def main():
    pin=json.loads(REGISTRY.read_text())['models'].get('robot-access-resume-v1')
    if pin is None:raise ValueError('Exact source approval must be registered first')
    manifest=verify(ROOT,contained(ROOT,pin['manifest']),pin['sha256']);cfg=manifest['runner_config']
    out=ROOT/'results/robot_access_resume_v1'
    if out.exists():raise ValueError('Prior output exists; no automatic retry/resume')
    out.parent.mkdir(exist_ok=True)
    with out.with_suffix('.stdout.log').open('xb') as stdout,out.with_suffix('.stderr.log').open('xb') as stderr:
        process=subprocess.Popen([cfg['learner_python'],'-I',str(ROOT/'experiments/source_consistency.py'),
            'run','--model-id',manifest['model_id'],'--output',str(out)],cwd=ROOT,
            stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        record=dict(pid=process.pid,manifest_sha256=pin['sha256'],started_unix=time.time())
        out.with_suffix('.launch.json').write_text(json.dumps(record,indent=2)+'\n')
        try:
            result=process.wait(timeout=cfg['development_wall_seconds']+cfg['final_wall_seconds']+300)
            record.update(status='exited',returncode=result)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            if process.poll() is None:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],check=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
                process.wait(timeout=30)
            record.update(status='terminated_by_watchdog',returncode=process.returncode)
        record['finished_unix']=time.time()
        out.with_suffix('.exit.json').write_text(json.dumps(record,indent=2)+'\n')
        if record['returncode']!=0:raise SystemExit(record['returncode'] or 1)


if __name__=='__main__':main()
