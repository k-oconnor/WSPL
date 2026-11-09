"""Read-only saved-artifact analysis; no model inference, fitting or author calls."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT.parent/'Skillforge-joint-attention-v1'
sys.path.insert(0,str(ARCHIVE/'experiments'))
from source_consistency import verify


def main():
    out=ARCHIVE/'results/joint_attention_v1'
    result=json.loads((out/'results.json').read_text())
    manifest=verify(ARCHIVE,ARCHIVE/'experiments/source_pins/joint_attention_v1.manifest.json',result['manifest_sha256'])
    assert hashlib.sha256((out/'author/final.npz').read_bytes()).hexdigest()==result['final_sha256']
    def bank(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    ref=bank(out/'reference.npz');final=bank(out/'author/final.npz')
    # Independently reconstruct the final two outer products from the published
    # turn-04 equations, then its FP32 storage and turn-05 normalization.
    W0=ref['W27'].astype(float);Xa=ref['asp_o27_input'].astype(float);Xb=ref['base_o27_input'].astype(float)
    A=ref['asp_layer27_out'].astype(float);B=ref['base_layer27_out'].astype(float)
    E=ref['answer_embeddings'].astype(float);g=ref['final_norm'].astype(float);X=.5*Xa+.5*Xb
    F=(E-E.mean(0,keepdims=True))*g
    _,s,Rt=np.linalg.svd(F,full_matrices=False)
    restoration=B*(np.linalg.norm(A,axis=1)/np.linalg.norm(B,axis=1))[:,None]-A
    projected=((restoration@Rt.T)*(s*s/(s*s+8.)))@Rt
    repair_direction=projected[[16,18]].mean(0);repair_direction/=np.linalg.norm(repair_direction)
    repair_gate=np.linalg.lstsq(np.stack([X[16],X[18],Xa[8],Xb[8]]),np.array([1.,1.,0.,0.]),rcond=1e-8)[0]
    repair=np.outer(repair_direction,repair_gate);repair*=5.30/np.linalg.norm(repair)
    H=B/np.sqrt(np.mean(B*B,axis=1,keepdims=True));common=H.mean(0)
    support_direction=(2*H[0]+H[8])/3-H[[2,4,6]].mean(0)
    support_direction-=common*(support_direction@common)/(common@common);support_direction/=np.linalg.norm(support_direction)
    support_gate=np.linalg.lstsq(np.stack([X[0],X[16],X[18],Xa[8],Xb[8]]),np.array([.5,0.,0.,1.,1.]),rcond=1e-8)[0]
    support=np.outer(support_direction,support_gate);support*=1.8/np.linalg.norm(support)
    d=repair+support;d*=.0495*np.linalg.norm(W0)/np.linalg.norm(d)
    stored=(W0+d).astype(np.float32)
    d=stored.astype(float)-W0;d*=.0499*np.linalg.norm(W0)/np.linalg.norm(d)
    reconstructed=(W0+d).astype(np.float32)
    assert np.array_equal(reconstructed,final['W27'])
    delta=final['W27'].astype(float)-W0;singular=np.linalg.svd(delta,compute_uv=False)
    variants=('untouched','initial_asp','joint','final_wsp_only','final_asp_only')
    records={name:{r['id']:r for r in json.loads((out/(name+'_reserved.json')).read_text())} for name in variants}
    rows=json.loads(Path(manifest['runner_config']['reserved']).read_text());acquired=json.loads(Path(manifest['runner_config']['acquisition']).read_text())
    word_ids=[row['id'] for row in rows if row['group'].startswith('word_')]
    shifts={}
    for name in variants[1:]:
        a=np.array([np.array(records[name][i]['normalized_logprobs'])-records['untouched'][i]['normalized_logprobs'] for i in word_ids])
        a-=a[:,1,None]  # Pairwise changes eliminate the common softmax normalizer.
        shifts[name]=dict(order=['conduction','convection','radiation','evaporation'],mean=a.mean(0).tolist(),std=a.std(0).tolist(),min=a.min(0).tolist(),max=a.max(0).tolist())
    transitions=[]
    for row in rows:
        if row['group']=='retention':continue
        entry=dict(id=row['id'],question=row['question'],group=row['group'],variants={})
        for name in variants:
            x=records[name][row['id']];scores=np.asarray(x['normalized_logprobs']);gold=x['gold']
            entry['variants'][name]=dict(prediction=row['choices']['text'][x['prediction']],correct=x['correct'],gold_margin=float(scores[gold]-np.max(np.delete(scores,gold))),scores=scores.tolist())
        transitions.append(entry)
    local={}
    for name in ('base','asp'):
        x=ref[name+'_o27_input'].astype(float)
        ratios=np.linalg.norm(x@delta.T,axis=1)/np.linalg.norm(x@W0.T,axis=1)
        local[name]=dict(min=float(ratios.min()),max=float(ratios.max()),mean=float(ratios.mean()),ratios=ratios.tolist())
    audit=dict(status='passed',read_only=True,new_model_forwards=0,new_author_calls=0,
               source_sha256=result['manifest_sha256'],final_sha256=result['final_sha256'],independent_weight_reconstruction_bit_identical=True,
               key_unchanged=np.array_equal(ref['K'],final['K']),top_singular_values=singular[:6].tolist(),top_two_energy=float(sum(singular[:2]**2)/sum(singular**2)),
               tail_frobenius=float(np.linalg.norm(singular[2:])),relative_weight_change=float(np.linalg.norm(delta)/np.linalg.norm(W0)),
               repair_anchors=[acquired[i] for i in (16,18)],relative_to_convection_logprob_shifts=shifts,local_projection_change=local,transitions=transitions)
    (ROOT/'docs/wspl/joint_attention_v1_investigation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('transitions','local_projection_change','relative_to_convection_logprob_shifts','repair_anchors')}))


if __name__=='__main__':main()
