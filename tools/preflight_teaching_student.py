"""Saved-student restoration and synthetic continuation checks; no task predictions."""
import gc
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from skillforge.authoring_sft import AuthoringSFTStudy,adapter_state,restore_adapter,frozen_hash,loss_sum,checkpoint
from skillforge.teaching_training import state_hash
from skillforge.reliable_json import save


def main():
    study=AuthoringSFTStudy('C:/Users/kevin/skillforge-runtime/models/Qwen2.5-Coder-1.5B-Instruct')
    model=study.model;base=frozen_hash(model)
    parent=torch.load(ROOT.parent/'WSPL-closed-loop-v1/results/compact_closed_loop_v1/seed_113/adapter_final.pt',map_location='cpu',weights_only=False)
    assert parent['base_hash']==base and parent['manifest_sha256']=='c5dacefd62c61ebcd9fcd439be9dca8e51ebda4620a65a7516f1aed6aefd90e3'
    restore_adapter(model,parent['adapter']);parent_hash=state_hash(parent['adapter'])
    assert state_hash(adapter_state(model))==parent_hash
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00003,weight_decay=.01,foreach=False)
    record=dict(ids=[100,200,300,400]*32,prefix=64,targets=64)
    def update():
        model.train();optimizer.zero_grad(set_to_none=True);loss=loss_sum(model,record)
        (loss/record['targets']).backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True);return float(loss.detach()),float(norm)
    update()
    with tempfile.TemporaryDirectory(prefix='teaching_restore_') as temp:
        p=Path(temp)/'checkpoint.pt';checkpoint(p,model,optimizer,dict(synthetic=True))
        expected_loss,_=update();expected=state_hash(adapter_state(model))
        saved=torch.load(p,map_location='cpu',weights_only=False);restore_adapter(model,saved['adapter']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        actual_loss,_=update();assert actual_loss==expected_loss and state_hash(adapter_state(model))==expected
        del saved
    # Stress a maximum-size prompt/backward without performing another optimizer step.
    long=dict(ids=[100,200,300,400]*1024,prefix=4000,targets=96)
    optimizer.zero_grad(set_to_none=True);loss=loss_sum(model,long)
    (loss/96).backward();assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache()
    assert frozen_hash(model)==base
    restore_adapter(model,parent['adapter']);assert state_hash(adapter_state(model))==parent_hash
    model.eval();model.gradient_checkpointing_disable()
    ids=torch.tensor([[100,200,300,400]],device='cuda')
    with torch.inference_mode():assert torch.isfinite(model(ids,use_cache=False).logits).all()
    result=dict(status='passed',synthetic_only=True,parent_adapter_sha256=parent_hash,base_hash=base,
        warm_start_exact=True,next_update_exact=True,parent_restored=True,base_unchanged=True,
        maximum_context_backward=4096,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    save(ROOT/'docs/wspl/teaching_student_preflight.json',result);print(json.dumps(result))


if __name__=='__main__':main()
