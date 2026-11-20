"""Isolated synthetic tensor update and camera/physics engineering checks."""
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.supervisor_workspace import sandbox_command,Budget,run_job,save


def main():
    runtime=Path(sys.argv[1]);python_home=Path(sys.base_prefix)
    with tempfile.TemporaryDirectory(prefix='supervisor_gpu_') as folder:
        folder=Path(folder);work=folder/'work';work.mkdir()
        (work/'check.py').write_text('''import sys,json
sys.path.insert(0,'/inputs/code')
import torch
assert torch.cuda.is_available()
torch.manual_seed(17)
m=torch.nn.Linear(4,2).cuda();o=torch.optim.AdamW(m.parameters(),lr=.01)
x=torch.ones(2,4,device='cuda');y=m(x);before=m.weight.detach().clone()
y.square().mean().backward();o.step()
assert not torch.equal(before,m.weight)
from skillforge.factory_observation import PandaBackend
b=PandaBackend(997,[[1,0,0],[0,1,0],[0,0,1]])
try:
 start=b.sensors()['timestamp_s'];b.step([0.,0.,0.,-1.])
 assert b.sensors()['timestamp_s']>start
 for camera in ('frontview','robot0_eye_in_hand'):
  frame=b.render(camera);assert frame.shape==(256,256,3)
finally:b.close()
print(json.dumps(dict(cuda=True,synthetic_update=True,physics_tick=True,cameras=2)))
''')
        b=Budget(folder/'ledger.json',180,120,1)
        args=sandbox_command(work,runtime,['/runtime/bin/python','/work/check.py'],
            readonly={'code':ROOT/'src'},gpu=True,python_home=python_home)
        result=run_job(b,'gpu_camera',args,folder/'logs',True,120)
        print((folder/'logs/stdout.log').read_text());print((folder/'logs/stderr.log').read_text())
        assert result['outcome']=='complete',result
        save(ROOT/'docs/wspl/supervisor_gpu_preflight.json',dict(status='passed',engineering_only=True,
            cuda_synthetic_update=True,cameras=2,physics_tick=True,model_calls=0,scientific_episodes=0))


if __name__=='__main__':main()
