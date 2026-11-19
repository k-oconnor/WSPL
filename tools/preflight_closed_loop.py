"""Synthetic-only full-size LoRA/optimizer/restoration preflight."""
import json
import sys
import tempfile
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
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
    prior_path=ROOT.parent/'WSPL-executive-v2/results/neural_executive_v2/adapter_final.pt'
    prior=torch.load(prior_path,map_location='cpu',weights_only=False)
    assert prior['base_hash']==base
    assert prior['manifest_sha256']=='43dabbb5b31820586f368de2037e2673767c71eafa89c55833d2d1fa6b60f5fc'
    restore_adapter(model,prior['adapter'])
    assert all(torch.equal(v,prior['adapter'][k]) for k,v in adapter_state(model).items())
    assert frozen_hash(model)==base
    report['previous_adapter_exact_restoration']=True
    from run_closed_loop import reset_adapter
    reset_adapter(model,113);fresh=adapter_state(model)
    assert all(not bool(v.any()) for k,v in fresh.items() if 'lora_B' in k)
    reset_adapter(model,227);different=adapter_state(model)
    assert any(not torch.equal(v,different[k]) for k,v in fresh.items() if 'lora_A' in k)
    reset_adapter(model,113)
    reset_state=adapter_state(model)
    assert all(torch.equal(v,reset_state[k]) for k,v in fresh.items())
    del reset_state
    report['full_size_seed_reset_exact']=True
    del fresh,different
    model.gradient_checkpointing_disable();model.eval()
    with torch.inference_mode():
        long_ids=torch.tensor([[100,200,300,400]*1900],device='cuda')
        generated=model.generate(long_ids,attention_mask=torch.ones_like(long_ids),do_sample=False,use_cache=True,max_new_tokens=4,pad_token_id=study.tokenizer.eos_token_id)
        assert generated.shape[1]>long_ids.shape[1]
    report['synthetic_long_prefill_tokens']=7600
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    assert frozen_hash(model)==base
    (ROOT/'docs/wspl/closed_loop_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
