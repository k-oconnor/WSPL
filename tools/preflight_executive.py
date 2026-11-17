"""Synthetic-only full-size LoRA/optimizer/restoration preflight."""
import json
import sys
import tempfile
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from skillforge.authoring_sft import AuthoringSFTStudy,LoRALinear,adapter_state,restore_adapter,frozen_hash,loss_sum,checkpoint


def main():
    torch.set_num_threads(4)
    study=AuthoringSFTStudy('C:/Users/kevin/skillforge-runtime/models/Qwen2.5-Coder-1.5B-Instruct')
    model=study.model;model.eval();base=frozen_hash(model)
    ids=torch.tensor([[100,200,300,400,500]],device='cuda')
    with torch.no_grad():
        wrapped=model(ids).logits.clone()
        modules=[m for m in model.modules() if isinstance(m,LoRALinear)]
        for m in modules:m.forward=m.base.forward
        direct=model(ids).logits.clone()
        for m in modules:del m.forward
    assert torch.equal(wrapped,direct),'zero adapter full-model parity'
    del wrapped,direct
    model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01,foreach=False)
    synthetic=dict(ids=([100,200,300,400]*512),prefix=512,targets=1536)
    torch.cuda.reset_peak_memory_stats();times=[]
    def update():
        torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
        loss=loss_sum(model,synthetic)/synthetic['targets'];assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();times.append(time.perf_counter()-start)
        return float(loss.detach()),float(norm)
    first=update()
    assert any(bool(p.any()) for n,p in model.named_parameters() if 'lora_B' in n)
    with tempfile.TemporaryDirectory(prefix='author-sft-preflight-') as temp:
        path=Path(temp)/'rolling.pt';checkpoint(path,model,opt,dict(step=1,synthetic=True))
        before=adapter_state(model);second=update();expected=adapter_state(model)
        saved=torch.load(path,weights_only=False);restore_adapter(model,saved['adapter']);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        assert all(torch.equal(before[n],p) for n,p in adapter_state(model).items())
        repeated=update();actual=adapter_state(model)
        assert second==repeated and all(torch.equal(expected[n],actual[n]) for n in expected),'next update differs after restoration'
    assert frozen_hash(model)==base,'base changed'
    report=dict(status='passed',synthetic_only=True,zero_adapter_logits_bit_identical=True,
                frozen_base_unchanged=True,next_update_exact_restoration=True,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                base_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                projections=len(study.projections),synthetic_tokens=2048,synthetic_completion_tokens=1536,
                update_seconds=times,peak_allocated_bytes=torch.cuda.max_memory_allocated(),first=first,second=second)
    prior_path=ROOT.parent/'WSPL-neural-dsl-v1/results/neural_executive_v1/adapter_final.pt'
    prior=torch.load(prior_path,map_location='cpu',weights_only=False)
    assert prior['base_hash']==base
    assert prior['manifest_sha256']=='9bde732531c5d8a939c87c7b224fefd52c9d4ee81ea4dcfba39adef08c4a09ef'
    restore_adapter(model,prior['adapter'])
    assert all(torch.equal(v,prior['adapter'][k]) for k,v in adapter_state(model).items())
    assert frozen_hash(model)==base
    report['previous_adapter_exact_restoration']=True
    (ROOT/'docs/wspl/executive_v2_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
