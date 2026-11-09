"""Synthetic repair diagnostics and construction checks; no benchmark cases."""
import json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import numpy as np
import torch
from skillforge.grid_movement import CopyFoundation
from skillforge.grid_repair import weights,load_weights,diagnostics,evaluate,validate
from skillforge.relu_adapter import ResidualReLU
from skillforge.arc_eval import model_hash
from skillforge.arc_authoring import read_bank
from run_attention_probe import construct


def main():
    previous=json.loads((ROOT/'docs/wspl/grid_movement_preflight.json').read_text());s=CopyFoundation(**previous['constructor_kwargs']);before=model_hash(s.model)
    parent=ROOT.parent/'Skillforge-grid-movement-v1/results/grid_movement_v1/last.pt'
    state=torch.load(parent,map_location='cpu',weights_only=False);a=ResidualReLU().to('cuda');a.load_state_dict(state['model']);a.eval().requires_grad_(False)
    original=weights(a);row=dict(input=[0,1,2,3]*4,output=[3,2,1,0]*4,task='movement');rows=[row,dict(row,task='copy')];demos=[row]*8
    diag=diagnostics(s,a,rows,demos);assert diag['H'].shape==(2,16,576) and diag['digit_logits'].shape==(2,16,4)
    assert all(np.isfinite(x).all() for x in diag.values());bank={**original,**diag};validate(bank,bank)
    first=evaluate(s,a,rows,demos)
    with tempfile.TemporaryDirectory(prefix='grid-repair-check-') as t:
        p=Path(t)/'parent.npz';np.savez_compressed(p,**bank);turn=Path(t)/'turn';turn.mkdir()
        candidate,feedback=construct("print({k:SHAPES[k] for k in NAMES})\nsave('write.bias', load('write.bias'))",p,turn,
            dict(python_worker=str(ROOT/'tools/arc_python_worker.py'),python_runtime='/home/kevin/venvs/skillforge-author'))
        assert feedback['exit_code']==0,feedback
        restored=read_bank(candidate,bank);validate(restored,bank);load_weights(a,restored)
        assert evaluate(s,a,rows,demos)==first
    assert model_hash(s.model)==before and model_hash(a)==model_hash_from_bank(original)
    report=dict(status='passed',synthetic_only=True,constructor_kwargs=previous['constructor_kwargs'],adapter_restore_exact=True,
        diagnostic_shapes={k:list(v.shape) for k,v in diag.items()},sandbox_noop_exact=True,foundation_unchanged=True,domain_predictions=0,author_calls=0,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (ROOT/'docs/wspl/grid_repair_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


def model_hash_from_bank(bank):
    import hashlib
    h=hashlib.sha256()
    for k,v in bank.items():h.update(k.encode());h.update(v.tobytes())
    return h.hexdigest()

if __name__=='__main__':main()
