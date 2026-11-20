"""Full-size isolated Qwen restoration and reversible synthetic update."""
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.supervisor_workspace import sandbox_command,Budget,run_job,save


def main():
    runtime=Path(sys.argv[1]);model=Path(sys.argv[2]);parent=Path(sys.argv[3])
    with tempfile.TemporaryDirectory(prefix='supervisor_student_') as folder:
        folder=Path(folder);work=folder/'work';work.mkdir()
        (work/'check.py').write_text('''import sys,json
sys.path.insert(0,'/inputs/code')
import torch
from skillforge.supervisor_workbench import StudentLab
from skillforge.authoring_sft import adapter_state,restore_adapter,frozen_hash
from skillforge.teaching_training import state_hash
s=StudentLab('/inputs/model','/inputs/parent')
saved=torch.load('/inputs/parent',map_location='cpu',weights_only=False)
assert state_hash(adapter_state(s.model))==state_hash(saved['adapter'])
assert frozen_hash(s.model)==saved['base_hash']
ids=torch.tensor([[1,2,3,4]],device='cuda')
s.model.eval()
with torch.no_grad():before=s.model(ids,use_cache=False).logits.detach().clone()
optimizer=torch.optim.AdamW(s.parameters(),lr=1e-5)
loss=s.model(ids,use_cache=False).logits.float().square().mean()
loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
assert state_hash(adapter_state(s.model))!=state_hash(saved['adapter'])
restore_adapter(s.model,saved['adapter'])
with torch.no_grad():after=s.model(ids,use_cache=False).logits
assert torch.equal(before,after)
assert frozen_hash(s.model)==saved['base_hash']
print(json.dumps(dict(parent_restored=True,synthetic_update=True,exact_logits_restored=True,
base_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated())))
''')
        b=Budget(folder/'ledger.json',600,600,1)
        args=sandbox_command(work,runtime,['/runtime/bin/python','/work/check.py'],
            readonly={'code':ROOT/'src','model':model,'parent':parent},gpu=True,python_home=sys.base_prefix)
        result=run_job(b,'student',args,folder/'logs',True,540)
        stdout=(folder/'logs/stdout.log').read_text();print(stdout);print((folder/'logs/stderr.log').read_text())
        assert result['outcome']=='complete',result
        report=json.loads(stdout.strip().splitlines()[-1]);report.update(status='passed',engineering_only=True)
        save(ROOT/'docs/wspl/supervisor_student_preflight.json',report)


if __name__=='__main__':main()
