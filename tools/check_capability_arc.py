"""Synthetic full-size restoration and scoring checks; no benchmark predictions."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import torch
from skillforge.arc_eval import ArcBaseline,model_hash,score_row,score_encoded,encode_choices
from skillforge.capability_bridge import modules,attached
from skillforge.go_replay import sha
from run_capability_arc import restore_foundation


def main():
    started=time.monotonic()
    parent=ROOT.parent/'Skillforge-capability6-v2/results/capability6_v1'
    kw=json.loads((ROOT/'docs/wspl/capability6_preflight.json').read_text())['constructor_kwargs']
    study=ArcBaseline(kw['model_path']);original=model_hash(study.model)
    restore_foundation(study,kw);before=model_hash(study.model)
    row=dict(id='synthetic',question='Choose a label for this synthetic scoring check.',
        choices=dict(label=['A','B'],text=['alpha','beta label']),answerKey='A')
    baseline=score_row(study.model,study.tokenizer,row)
    frozen=json.loads((parent/'frozen.json').read_text());checks={}
    for arm,h in frozen.items():
        path=parent/arm/'final.json';assert sha(path)==h
        plan=json.loads(path.read_text());bank=modules(plan).eval().requires_grad_(False)
        state=torch.load(parent/arm/'last.pt',map_location='cpu',weights_only=False)
        for name,value in bank.state_dict().items():
            assert torch.equal(value.cpu(),state['modules'][name]),(arm,name)
        bank_hash=model_hash(bank)
        t=time.monotonic()
        with attached(study.model,bank):
            score=score_row(study.model,study.tokenizer,row)
            encoded=encode_choices(study.tokenizer,row)
            single=[score_encoded(study.model,[item],study.tokenizer.eos_token_id,'cuda')[0][0] for item in encoded]
        error=max(abs(a-b) for a,b in zip(score['answer_logprobs'],single))
        assert error<1e-4,(arm,error)
        assert score_row(study.model,study.tokenizer,row)==baseline
        assert model_hash(bank)==bank_hash and model_hash(study.model)==before
        checks[arm]=dict(final_sha256=h,checkpoint_tensor_parity=True,hook_removal_exact=True,
            batched_single_logprob_error=error,synthetic_scoring_seconds=time.monotonic()-t)
        del bank,state;torch.cuda.empty_cache()
    report=dict(status='passed',synthetic_only=True,original_model_sha256=original,foundation=kw,arms=checks,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,benchmark_predictions=0)
    (ROOT/'docs/wspl/capability_arc_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
