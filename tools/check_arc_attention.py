"""Synthetic full-model boundary-scoring checks; no ARC predictions."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from skillforge.arc_eval import ArcBaseline,score_row,model_hash
from skillforge.arc_attention import score_boundary
from skillforge.attention_probe import prefix_score


def main():
    s=ArcBaseline('C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct');before=model_hash(s.model)
    path=ROOT.parent/'Skillforge-attention-probe-resume-v1/results/attention_probe_resume_v1/retrieval/final.npz'
    with np.load(path,allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
    row=dict(id='synthetic',question='Synthetic fixture: red blue green.',choices=dict(label=['A','B'],text=['red blue green','green red blue']),answerKey='A')
    baseline=score_row(s.model,s.tokenizer,row);off=score_boundary(s,row)
    error=max(abs(a-c) for a,c in zip(baseline['answer_logprobs'],off['answer_logprobs']));assert error<2e-5,error
    edited=score_boundary(s,row,b,'retrieval')
    masses=edited['virtual_mass_by_choice']
    assert np.max(np.abs(np.asarray(masses[0][0]['mass'])-np.asarray(masses[1][0]['mass'])))<2e-6
    one={**row,'choices':dict(label=['A','B'],text=['red','blue'])}
    a=prefix_score(s,one,b,'retrieval');c=score_boundary(s,one,b,'retrieval')
    single_error=max(abs(x-y) for x,y in zip(a['logprobs'],c['answer_logprobs']));assert single_error<2e-5
    assert score_boundary(s,row)==off and model_hash(s.model)==before
    result=dict(status='passed',synthetic_only=True,disabled_error=error,single_token_error=single_error,
                routing_answer_independent=True,base_restored=True,benchmark_predictions=0,model_sha256=before)
    (ROOT/'docs/wspl/arc_attention_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
