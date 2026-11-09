"""Full-model synthetic feature/mass validation plus augmented-bank sandbox check."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import numpy as np
import torch
from skillforge.arc_eval import ArcBaseline,model_hash
from skillforge.attention_probe import prefix_score
from skillforge.selective_key import features,validate_key
from skillforge.arc_authoring import read_bank
from run_attention_probe import construct


def main():
    study=ArcBaseline('C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct');before=model_hash(study.model)
    choices=dict(label=['A','B','C','D'],text=['yes','no','red','blue'])
    rows=[dict(id='synthetic_'+str(i),question='alpha beta '+str(i),choices=choices,answerKey='A') for i in range(13)]
    q,logz,base=features(study,rows)
    parent=ROOT.parent/'Skillforge-attention-probe-resume-v1/results/attention_probe_resume_v1/retrieval/final.npz'
    with np.load(parent,allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    bank.update(train_Q=q,train_logZ=logz,train_applicable=np.zeros(13,np.float32))
    actual=[]
    for row in rows:actual.append(prefix_score(study,row,bank,'retrieval')['virtual_mass'][0]['mass'][0])
    predicted=1/(1+np.exp(-(q.astype(np.float64)@bank['K'][2,0].astype(np.float64)/8-logz)))
    error=float(np.max(np.abs(predicted-np.array(actual))));assert error<2e-5,error
    for row,expected in zip(rows,base):assert prefix_score(study,row,bank,'off')==expected
    assert model_hash(study.model)==before
    folder=ROOT/'work'/('selective_key_check_'+str(time.time_ns()));folder.mkdir(parents=True)
    np.savez_compressed(folder/'parent.npz',**bank)
    path,feedback=construct("k=load('K'); k[2,0]=0; save('K',k); print(load('train_Q').shape)",folder/'parent.npz',folder,
        dict(python_worker=str(ROOT/'tools/arc_python_worker.py'),python_runtime='/home/kevin/venvs/skillforge-author'))
    assert feedback['exit_code']==0,feedback
    candidate=read_bank(path,bank);validate_key(candidate,bank)
    assert not np.any(candidate['K'][2,0])
    result=dict(status='passed',synthetic_only=True,diagnostic_mass_max_error=error,model_sha256=before,
        restored_scores_exact=True,augmented_bank_sandbox_passed=True,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),author_calls=0,benchmark_predictions=0)
    (ROOT/'docs/wspl/selective_key_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
