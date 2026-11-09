"""Synthetic full-size key fit and frozen-model insertion/restoration; no benchmark."""
import json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import numpy as np
import torch
from skillforge.key_learning import fit_key,DEFAULT
from skillforge.arc_eval import ArcBaseline,model_hash
from skillforge.attention_probe import intervention
from skillforge.selective_key import validate_key


def main():
    study=ArcBaseline('C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct');before=model_hash(study.model)
    parent=ROOT.parent/'Skillforge-attention-probe-resume-v1/results/attention_probe_resume_v1/retrieval/final.npz'
    with np.load(parent,allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    rng=np.random.default_rng(761);q=rng.normal(size=(1024,3,64));y=np.arange(1024)%4==0
    q[:,:,0]+=np.where(y[:,None],2.,-2.);z=rng.normal(size=(1024,3));correct=np.arange(1024)%3==0
    t=time.monotonic();key,trace=fit_key(q,z,y,correct,bank['K'][2,0],10*float(bank['K_scale'][2,0]),DEFAULT)
    fit_seconds=time.monotonic()-t;candidate={k:v.copy() for k,v in bank.items()};candidate['K'][2,0]=key;validate_key(candidate,bank)
    ids=torch.tensor([[1,137,81,924,55,238,2]],device='cuda')
    def forward():
        with torch.inference_mode():return study.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
    original=forward()
    with intervention(candidate,'retrieval'):edited=forward()
    assert torch.isfinite(edited).all() and torch.equal(original[:,:-1],edited[:,:-1])
    assert torch.equal(original,forward()) and model_hash(study.model)==before
    result=dict(status='passed',synthetic_only=True,examples=1024,optimizer_steps=1000,fit_seconds=fit_seconds,
                key_finite=True,norm=float(np.linalg.norm(key)),base_restored_exact=True,model_sha256=before,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),benchmark_predictions=0,author_calls=0)
    (ROOT/'docs/wspl/key_learning_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
