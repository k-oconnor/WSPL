"""Full-size synthetic checks only; no capability benchmark or author calls."""
import sys,json,tempfile,random,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from skillforge.grid_movement import CopyFoundation
from skillforge.arc_eval import model_hash
from skillforge.capability_bridge import default_plan,modules,attached,export,validate_plan
from skillforge.capability_codec import encode,generate
from skillforge.grid_copy import batch_tensors
from skillforge.capability_training import loss,save_checkpoint
from run_capability6 import construct


def main():
    start=time.monotonic();kw=json.loads((ROOT/'docs/wspl/grid_movement_preflight.json').read_text())['constructor_kwargs'];s=CopyFoundation(**kw);before=model_hash(s.model)
    plan=default_plan();meta=validate_plan(plan);bank=modules(plan)
    # Arbitrary synthetic arrays, not valid movement fixtures or benchmark cases.
    row=dict(size=6,task='move',input=([0,1,2,3,4,5]*6),target=([5,4,3,2,1,0]*6));encoded=[encode(s.tokenizer,row)]*4
    batch=batch_tensors(encoded,s.tokenizer.eos_token_id,'cuda')
    with torch.no_grad():original=s.model(input_ids=batch[0],attention_mask=batch[1],use_cache=False).logits
    opt=torch.optim.AdamW(bank.parameters(),lr=.001,weight_decay=0,foreach=False)
    with attached(s.model,bank):
        with torch.no_grad():assert torch.equal(original,s.model(input_ids=batch[0],attention_mask=batch[1],use_cache=False).logits)
        del original;torch.cuda.synchronize();t=time.monotonic();value=loss(s.model,batch,[1.]*144);value.backward();torch.nn.utils.clip_grad_norm_(bank.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();update_seconds=time.monotonic()-t
        assert torch.isfinite(value);cached=generate(s,row,True);uncached=generate(s,row,False);assert cached['tokens']==uncached['tokens']
        with torch.no_grad():expected=s.model(input_ids=batch[0],attention_mask=batch[1],use_cache=False).logits.detach().cpu()
    trained=export(plan,bank)
    with tempfile.TemporaryDirectory(prefix='capability6-check-') as temp:
        p=Path(temp)/'last.pt';save_checkpoint(p,bank,opt,dict(updates=1,train_seconds=update_seconds,targets=144),'synthetic',random.Random(1),{k:m.graph for k,m in bank.items()})
        state=torch.load(p,map_location='cpu',weights_only=False);restored=modules(trained);restored.load_state_dict(state['modules'])
        with attached(s.model,restored),torch.no_grad():assert torch.equal(expected,s.model(input_ids=batch[0],attention_mask=batch[1],use_cache=False).logits.cpu())
        # Minimal actual native/WSPL sandbox construction, through the production worker.
        small=dict(recipe=plan['recipe'],branches={'0_q':plan['branches']['0_q']})
        config=dict(worker=str(ROOT/'tools/capability_worker.py'),python_runtime='/home/kevin/venvs/skillforge-author')
        results=[]
        for interface,source in [('native',"g,a=load('0_q'); a['b'][:]=.25; save('0_q',g,a)"),('wspl',"wspl('0_q', 'wspl 0.2; target base=artifact(\"base\",sha256=\"$PARENT\"); edit base as candidate mode=fixed { b[:]=1/4; }')")]:
            folder=Path(temp)/interface;folder.mkdir();candidate,feedback=construct(source,small,{},interface,folder,config)
            assert candidate is not None,feedback;results.append(candidate)
        assert results[0]==results[1]
    assert all(p.grad is None for p in s.model.parameters()) and model_hash(s.model)==before
    report=dict(status='passed',synthetic_only=True,constructor_kwargs=kw,parameters=meta['parameters'],branches=len(bank),zero_parity_exact=True,
        cached_generation_tokens_equal=True,full_checkpoint_logits_exact=True,sandbox_native_wspledit_exact=True,base_unchanged=True,
        update_seconds=update_seconds,sequence_tokens=batch[0].shape[1],peak_allocated_bytes=torch.cuda.max_memory_allocated(),total_seconds=time.monotonic()-start,benchmark_predictions=0,author_calls=0)
    (ROOT/'docs/wspl/capability6_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
