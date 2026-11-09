"""Synthetic integrated-program and sandbox preflight; no benchmark predictions."""
import sys,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import numpy as np
from skillforge.arc_eval import ArcBaseline,model_hash
from skillforge.arc_attention import score_boundary
from skillforge.joint_attention import make_reference,validate,weight_edit
from skillforge.relational_program import NeuralProgram
from skillforge.relational_program import make_program_reference as make_reference
from skillforge.arc_authoring import read_bank
from run_attention_probe import construct


def main():
    s=ArcBaseline('C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct');before=model_hash(s.model)
    path=ROOT.parent/'Skillforge-attention-probe-resume-v1/results/attention_probe_resume_v1/retrieval/final.npz'
    with np.load(path,allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
    row=dict(id='synthetic',question='Synthetic fixture: red blue green.',choices=dict(label=['A','B'],text=['red blue green','green red blue']),answerKey='A',group='fixture')
    bank,base,initial=make_reference(s,parent,[row]);assert bank['base_o27_input'].shape==(1,576)
    assert bank['acquisition_Q'].shape==(1,3,64) and bank['acquisition_logZ'].shape==(1,3)
    logits=bank['acquisition_Q'][0].astype(float)@bank['K'][2,0].astype(float)/8-bank['acquisition_logZ'][0]
    predicted=1/(1+np.exp(-logits));measured=np.asarray(initial[0]['virtual_mass_by_choice'][0][0]['mass'][0])
    assert np.max(np.abs(predicted-measured))<2e-6
    edit={k:v.copy() for k,v in bank.items()};edit['W27']*=1.001;edit['K'][2,0]*=.99;edit['V'][2,0]*=.99;validate(edit,bank)
    with weight_edit(s.model,edit):scored=score_boundary(s,row,edit,'retrieval')
    assert all(np.isfinite(scored['answer_logprobs']))
    compact={k:edit[k].copy() for k in ('K','V','alpha','W27')}
    program=NeuralProgram(compact)
    assert program.score(s,row)==scored
    changed_gold={**row,'answerKey':'B'}
    assert program.score(s,changed_gold)['answer_logprobs']==scored['answer_logprobs']
    for variant in ('untouched','wsp_only','asp_only'):
        assert np.isfinite(program.score(s,row,variant)['answer_logprobs']).all()
    assert model_hash(s.model)==before and score_boundary(s,row)==base[0]
    with tempfile.TemporaryDirectory(prefix='joint-preflight-') as t:
        folder=Path(t);p=folder/'parent.npz';np.savez_compressed(p,**bank)
        candidate,feedback=construct("w=load('W27'); save('W27',w*1.001)",p,folder,dict(python_worker=str(ROOT/'tools/arc_python_worker.py'),python_runtime='/home/kevin/venvs/skillforge-author'))
        assert feedback['exit_code']==0,feedback
        constructed=read_bank(candidate,bank);validate(constructed,bank)
        assert np.array_equal(constructed['W27'],bank['W27']*np.float32(1.001))
    result=dict(status='passed',synthetic_only=True,full_model_restoration=True,diagnostics=True,sandbox=True,
                model_sha256=before,benchmark_predictions=0,author_calls=0)
    (ROOT/'docs/wspl/relational_program_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
