"""Full-size synthetic-token interface check. No ARC predictions or author calls."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from skillforge.arc_eval import ArcBaseline,model_hash
from skillforge.attention_probe import intervention


def main():
    study=ArcBaseline('C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct')
    before=model_hash(study.model);ids=torch.tensor([[1,137,81,924,55,238,2]],device='cuda')
    def forward():
        with torch.inference_mode():return study.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
    original=forward();capture={}
    with intervention(capture=capture):captured=forward()
    assert torch.equal(original,captured) and len(capture['Q'])==6
    b=dict(K=np.zeros((6,3,64),np.float32),V=np.zeros((6,3,64),np.float32),alpha=np.zeros((6,3),np.float32))
    with intervention(b,'forced'):noop=forward()
    assert torch.equal(original,noop)
    b['alpha'][-1,0]=.5;b['V'][-1,0]=capture['native_V'][-1][0,0]
    with intervention(b,'forced'):forced=forward()
    b['K'][-1,0]=capture['Q'][-1][0];mass=[]
    with intervention(b,'retrieval',masses=mass):retrieved=forward()
    assert torch.isfinite(forced).all() and torch.isfinite(retrieved).all()
    assert torch.equal(original[:,:-1],forced[:,:-1]) and torch.equal(original[:,:-1],retrieved[:,:-1])
    assert not torch.equal(original,forced) and not torch.equal(original,retrieved)
    assert torch.equal(original,forward()) and model_hash(study.model)==before
    result=dict(status='passed',synthetic_only=True,benchmark_predictions=0,author_calls=0,
                no_op_bit_identical=True,earlier_positions_bit_identical=True,restored_logits_bit_identical=True,
                model_sha256=before,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),retrieval_mass=mass)
    path=ROOT/'docs/wspl/attention_probe_preflight.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
