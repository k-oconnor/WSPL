"""Full-size synthetic inference and checkpoint switching, no task predictions."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from skillforge.authoring_sft import AuthoringSFTStudy,adapter_state,restore_adapter,frozen_hash


def main():
    s=AuthoringSFTStudy('C:/Users/kevin/skillforge-runtime/models/Qwen2.5-Coder-1.5B-Instruct')
    m=s.model;m.eval();m.gradient_checkpointing_disable();base=frozen_hash(m)
    parent=ROOT.parent/'WSPL-closed-loop-v1/results/compact_closed_loop_v1'
    paths=[parent/f'seed_{seed}/adapter_final.pt' for seed in [113,227,341]]
    paths.append(ROOT.parent/'WSPL-controller-author-v1/results/controller_author_v1/adapter_final.pt')
    ids=torch.tensor([[100,200,300,400]],device='cuda');first=None
    for i,p in enumerate(paths+[paths[0]]):
        ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['base_hash']==base
        restore_adapter(m,ck['adapter']);actual=adapter_state(m)
        assert all(torch.equal(v,actual[k]) for k,v in ck['adapter'].items())
        with torch.inference_mode():
            logits=m(ids,use_cache=False).logits
            assert torch.isfinite(logits).all()
            if i==0:first=logits.clone()
            if i==4:assert torch.equal(first,logits)
        del ck,actual
    with torch.inference_mode():
        long=torch.tensor([[100,200,300,400]*1900],device='cuda')
        generated=m.generate(long,attention_mask=torch.ones_like(long),do_sample=False,use_cache=True,max_new_tokens=4,pad_token_id=s.tokenizer.eos_token_id)
        assert generated.shape[1]>long.shape[1]
    assert frozen_hash(m)==base
    r=dict(status='passed',synthetic_only=True,all_four_adapters_exact=True,round_trip_logits_exact=True,
           base_unchanged=True,long_prefill_tokens=7600,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (ROOT/'docs/wspl/astra_adaptation_preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))


if __name__=='__main__':main()
