"""Synthetic adapter step/restoration on the frozen copy foundation."""
import sys,json,tempfile,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from skillforge.grid_movement import CopyFoundation,encode_task
from skillforge.grid_copy import batch_tensors,copy_loss,checkpoint
from skillforge.relu_adapter import ResidualReLU,attached
from skillforge.arc_eval import model_hash

KW=dict(model_path='C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct',
    checkpoint_path=str(ROOT.parent/'Skillforge-grid-copy-v1/results/grid_copy_v1/last.pt'),
    checkpoint_sha256='633fb81d3bd911a525903bf0bcaa5dfaaffc3ea621f6cb0318a1f58b1eae27db',
    model_sha256='b8cf2a97f33aca737f43c3e439c1caac79f23bf793b0e9bd6ac8eb8dec776f38')


def main():
    s=CopyFoundation(**KW);before=model_hash(s.model);a=ResidualReLU().to('cuda')
    # Arbitrary synthetic arrays, not valid movement-domain cases.
    row=dict(input=[0,1,2,3]*4,output=[3,2,1,0]*4);demos=[row]*8
    batch=batch_tensors([encode_task(s.tokenizer,row,'movement',demos)]*3+[encode_task(s.tokenizer,row,'copy',demos)],s.tokenizer.eos_token_id,'cuda')
    with torch.no_grad():base=s.model(input_ids=batch[0],attention_mask=batch[1],use_cache=False).logits
    opt=torch.optim.AdamW(a.parameters(),lr=.001,weight_decay=0,foreach=False)
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'last.pt'
        with attached(s.model,a):
            with torch.no_grad():assert torch.equal(base,s.model(input_ids=batch[0],attention_mask=batch[1],use_cache=False).logits)
            del base
            loss=copy_loss(s.model,batch);loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
            assert torch.isfinite(loss);assert float(a.write.weight.norm())>0
            checkpoint(p,a,opt,1,'synthetic',[],random.Random(3));saved=torch.load(p,weights_only=False)
            restored=ResidualReLU().to('cuda');restored.load_state_dict(saved['model']);assert model_hash(a)==model_hash(restored)
        assert all(p.grad is None for p in s.model.parameters());assert model_hash(s.model)==before
    report=dict(status='passed',synthetic_only=True,constructor_kwargs=KW,adapter_parameters=sum(p.numel() for p in a.parameters()),zero_adapter_exact=True,
        frozen_base_unchanged=True,finite_adapter_update=True,adapter_restore_exact=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),sequence_tokens=batch[0].shape[1],domain_predictions=0)
    (ROOT/'docs/wspl/grid_movement_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
