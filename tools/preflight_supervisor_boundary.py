"""Linux-only engineering smoke: isolation, freeze/reload, timeout cleanup.

Synthetic policies only. No pretrained model, training run or robot benchmark.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.supervisor_workspace import Budget,freeze,verify_bundle,sandbox_command,run_job,save


def main():
    runtime=Path(sys.argv[1]);output=Path(sys.argv[2])
    with tempfile.TemporaryDirectory(prefix='supervisor_boundary_') as tmp:
        root=Path(tmp);work=root/'work';work.mkdir()
        secret=root/'sealed.txt';secret.write_text('sealed-evaluator-fixture')
        code="""import pathlib,socket
assert not pathlib.Path(SECRET).exists()
assert not pathlib.Path('/mnt/c').exists()
assert not pathlib.Path('/home/kevin').exists()
assert not pathlib.Path('/dev/dxg').exists()
s=socket.socket();s.settimeout(.2)
try:s.connect(('1.1.1.1',443))
except OSError:pass
else:raise AssertionError('network available')
print('isolated')
""".replace('SECRET',repr(str(secret)))
        (work/'check.py').write_text(code)
        budget=Budget(root/'ledger.json',120,60,2)
        args=sandbox_command(work,runtime,['/runtime/bin/python','/work/check.py'])
        result=run_job(budget,'boundary',args,root/'logs1',seconds=15)
        assert result['outcome']=='complete',(root/'logs1/stderr.log').read_text()
        # Scripted revision stands in for a supervisor mutating its policy.
        candidate=root/'candidate';candidate.mkdir()
        save(candidate/'submission.json',dict(entrypoint='policy.py',kind='synthetic'))
        (candidate/'policy.py').write_text('''class Policy:
 def __init__(self,root,declaration): self.count=0
 def reset(self,public): self.count=0
 def act(self,observation):
  self.count+=1
  return {"op":"probe","command":[0.1,0,0],"ticks":self.count}
 def close(self):pass
''')
        bundle=freeze(candidate,root/'store',dict(engineering_only=True));verify_bundle(bundle)
        worker=ROOT/'tools/supervisor_deployment_worker.py'
        args=sandbox_command(work,runtime,['/runtime/bin/python','/inputs/worker','/inputs/bundle'],
            readonly=dict(worker=worker,bundle=bundle))
        requests=[dict(op='reset',public={}),dict(op='act',observation={}),dict(op='reset',public={}),dict(op='act',observation={}),dict(op='close')]
        result=subprocess.run(args,input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,timeout=15)
        assert result.returncode==0,result.stderr
        replies=[json.loads(l) for l in result.stdout.splitlines()]
        assert replies[1]['result']['ticks']==replies[3]['result']['ticks']==1
        # Child must not survive sandbox timeout and write a marker later.
        (work/'timeout.py').write_text("import subprocess,time\nsubprocess.Popen(['/runtime/bin/python','-c',\"import time,pathlib;time.sleep(1);pathlib.Path('/work/leaked').write_text('bad')\"])\ntime.sleep(10)\n")
        args=sandbox_command(work,runtime,['/runtime/bin/python','/work/timeout.py'])
        result=run_job(budget,'timeout',args,root/'logs2',seconds=.25)
        assert result['outcome']=='timeout';time.sleep(1.2)
        assert not (work/'leaked').exists()
        report=dict(status='passed',engineering_only=True,host_files_hidden=True,network_disabled=True,
            cpu_job_gpu_hidden=True,bundle_verified=True,episode_state_reset=True,descendant_timeout_passed=True,
            model_calls=0,scientific_episodes=0)
        save(output,report);print(json.dumps(report))


if __name__=='__main__':main()
