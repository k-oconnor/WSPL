"""Engineering preflight only: synthetic token IDs, no ARC predictions."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
import numpy as np
from skillforge.arc_eval import ArcBaseline,model_hash,encode_choices,score_encoded
from skillforge.smollm_adapter import TARGETS,initial_bank,effective_weights


def main():
    started=time.monotonic();study=ArcBaseline('C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct')
    params=dict(study.model.named_parameters());base={k:params[name].detach().cpu().numpy().copy() for k,name in TARGETS.items()}
    before=model_hash(study.model)
    x=torch.tensor([[1,42,73,128]],device='cuda');mask=torch.ones_like(x)
    with torch.inference_mode():original=study.model(input_ids=x,attention_mask=mask,use_cache=False).logits.clone()
    merged,_=effective_weights(base,initial_bank(base,'lora'),'lora')
    with torch.no_grad():
        for k,name in TARGETS.items():params[name].copy_(torch.from_numpy(merged[k]).to('cuda'))
    with torch.inference_mode():after=study.model(input_ids=x,attention_mask=mask,use_cache=False).logits
    assert torch.isfinite(after).all() and torch.equal(original,after)
    assert before==model_hash(study.model)
    synthetic=[([1,42,73,128],2),([1,42,73],2)]
    batched,_=score_encoded(study.model,synthetic,study.tokenizer.eos_token_id,'cuda')
    single=[score_encoded(study.model,[row],study.tokenizer.eos_token_id,'cuda')[0][0] for row in synthetic]
    batch_error=max(abs(a-b) for a,b in zip(batched,single))
    assert batch_error<1e-4,batch_error
    lengths={}
    for task in ('challenge','easy'):
        rows=[json.loads(x) for x in (Path('C:/Users/kevin/skillforge-runtime/datasets/arc_baseline_v1')/(task+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        lengths[task]=max(len(ids) for row in rows for ids,_ in encode_choices(study.tokenizer,row))
    out=dict(status='passed',synthetic_forward_only=True,benchmark_predictions=0,unchanged_model_sha256=before,
        parameters=sum(p.numel() for p in study.model.parameters()),direct_edit_parameters=sum(v.size for v in base.values()),lora_parameters=4*8*(576+576),
        max_encoded_lengths=lengths,zero_adapter_logits_bit_identical=True,synthetic_batch_singleton_max_error=batch_error,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch=torch.__version__,seconds=time.monotonic()-started)
    (ROOT/'docs/wspl/smollm_bridge_preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
