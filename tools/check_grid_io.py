"""Synthetic codec/model/adapter checks, without scored domain examples."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from skillforge.arc_eval import ArcBaseline,model_hash
from skillforge.grid_io import generate,palette_ids
from skillforge.relu_adapter import ResidualReLU,attached


def main():
    s=ArcBaseline('C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct');before=model_hash(s.model)
    # This is an arbitrary format fixture, not a valid object/marker domain case.
    grid=[0,1,2,3]*4
    for mode in ('unconstrained','palette_only'):
        result=generate(s,grid,'copy',[],mode);assert len(result['tokens'])==16
        if mode=='palette_only':assert result['format_valid'] and len(result['output'])==16
    ids=torch.tensor([[32,33,34,35,32,33]],device='cuda')
    with torch.no_grad():base=s.model(input_ids=ids,use_cache=False).logits
    adapter=ResidualReLU().to('cuda');assert sum(x.numel() for x in adapter.parameters())==37472
    with attached(s.model,adapter):
        logits=s.model(input_ids=ids,use_cache=False).logits
        assert torch.equal(base,logits.detach())
        loss=torch.nn.functional.cross_entropy(logits[0,-1:,:],torch.tensor([32],device='cuda'));loss.backward()
    assert adapter.write.weight.grad is not None and float(adapter.write.weight.grad.norm())>0
    assert all(x.grad is None for x in s.model.parameters())
    with torch.no_grad():assert torch.equal(base,s.model(input_ids=ids,use_cache=False).logits)
    assert model_hash(s.model)==before
    result=dict(status='passed',synthetic_only=True,palette_ids=palette_ids(s.tokenizer),model_sha256=before,
                adapter_parameters=37472,zero_adapter_exact=True,adapter_gradient_connected=True,base_unchanged=True,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),domain_predictions=0,author_calls=0)
    (ROOT/'docs/wspl/grid_io_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
