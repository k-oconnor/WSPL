"""One synthetic optimizer step and restoration, no domain examples or scientific fitting."""
import sys,json,time,tempfile,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from skillforge.arc_eval import ArcBaseline,model_hash
from skillforge.grid_copy import encode_copy,batch_tensors,copy_loss,checkpoint


def main():
    start=time.monotonic();s=ArcBaseline('C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct');before=model_hash(s.model)
    # Arbitrary non-domain arrays and arbitrary copy demonstrations only.
    demos=[dict(input=([i%4]*16)) for i in range(8)]
    encoded=[encode_copy(s.tokenizer,[0,1,2,3]*4,demos)]*4
    batch=batch_tensors(encoded,s.tokenizer.eos_token_id,'cuda');s.model.requires_grad_(True).train()
    opt=torch.optim.AdamW(s.model.parameters(),lr=.00002,weight_decay=0,foreach=False)
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'initial.pt';checkpoint(p,s.model,opt,0,'synthetic',[],random.Random(1))
        loss=copy_loss(s.model,batch);assert torch.isfinite(loss);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(s.model.parameters(),1.,error_if_nonfinite=True);opt.step();assert model_hash(s.model)!=before
        opt.zero_grad(set_to_none=True);state=torch.load(p,map_location='cpu',weights_only=False);s.model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        assert model_hash(s.model)==before
    report=dict(status='passed',synthetic_only=True,model_sha256=before,finite_full_model_update=True,exact_restore=True,
        answer_targets=int((batch[2]!=-100).sum()),sequence_length=batch[0].shape[1],peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start)
    (ROOT/'docs/wspl/grid_copy_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
