"""Full-size synthetic bridge checks; no benchmark predictions or author calls."""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.arc_eval import ArcBaseline,model_hash
from skillforge.smollm_adapter import TARGETS,initial_bank,effective_weights
from skillforge.arc_authoring import read_bank
from run_arc_authoring import construct


def main():
    study=ArcBaseline('C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct')
    before=model_hash(study.model);params=dict(study.model.named_parameters())
    base={k:params[name].detach().cpu().numpy().copy() for k,name in TARGETS.items()}
    cfg=dict(python_worker=str(ROOT/'tools/arc_python_worker.py'),wspl_worker=str(ROOT/'tools/arc_wspl_worker.py'),python_runtime='/home/kevin/venvs/skillforge-author')
    template='wspl 0.2;target b=artifact("parent",sha256="{{PARENT_SHA256}}");edit b as c mode=fixed numeric=exact_to_storage {'
    sources=dict(native="w=load('W26');w[0,0]+=np.float32(1/1024);save('W26',w)",wspl=template+'W26[0,0]+=1/1024;}',lora=template+'A26[0,0]=1/32;B26[0,0]=1/32;}')
    outputs={};receipts={}
    with tempfile.TemporaryDirectory(dir=ROOT.parent) as d:
        folder=Path(d);bp=folder/'base.npz';np.savez_compressed(bp,**base)
        for arm,source in sources.items():
            mode='lora' if arm=='lora' else 'direct';bank=initial_bank(base,mode);parent=folder/(arm+'.npz');np.savez_compressed(parent,**bank)
            turn=folder/arm;turn.mkdir();p,feedback=construct(arm,source,parent,bp,turn,cfg)
            assert feedback['exit_code']==0,feedback
            restored=read_bank(p,bank);outputs[arm],metrics=effective_weights(base,restored,mode);receipts[arm]=dict(seconds=feedback['seconds'],metrics=metrics)
        for k in base:
            assert np.array_equal(outputs['native'][k],outputs['wspl'][k])
            assert np.array_equal(outputs['native'][k],outputs['lora'][k])
        x=torch.tensor([[1,42,73,128]],device='cuda');mask=torch.ones_like(x)
        for arm in outputs:
            with torch.no_grad():
                for k,name in TARGETS.items():params[name].copy_(torch.from_numpy(outputs[arm][k]).to('cuda'))
            with torch.inference_mode():logits=study.model(input_ids=x,attention_mask=mask,use_cache=False).logits
            assert torch.isfinite(logits).all()
            if arm=='native':reference=logits.clone()
            else:assert torch.equal(logits,reference)
        with torch.no_grad():
            for k,name in TARGETS.items():params[name].copy_(torch.from_numpy(base[k]).to('cuda'))
        assert model_hash(study.model)==before
    report=dict(status='passed',full_size=True,benchmark_predictions=0,author_calls=0,three_arm_effective_weights_bit_identical=True,
        three_arm_synthetic_logits_bit_identical=True,base_restoration_sha256=before,receipts=receipts,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (ROOT/'docs/wspl/arc_authoring_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
