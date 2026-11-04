"""Read-only completed diffusion editing audit; no authoring or sampled rollouts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

MAIN=Path(__file__).resolve().parents[1]
RUN=MAIN.parent/'Skillforge-diffusion-edit-v1'
sys.path[:0]=[str(RUN/'src'),str(RUN/'experiments')]
from source_consistency import verify
from skillforge.codex_one_shot import audit_events
from skillforge.diffusion_edit import linux_path,extract


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=RUN/'results/diffusion_edit_v1'
    r=json.loads((out/'results.json').read_text());receipt=json.loads((out/'source_receipt.json').read_text())
    m=verify(RUN,out/'approved_manifest.json','494c130f55b69d0705885e15ea5c3af4a49fe31883c4319ac870e8dec939a4ac')
    assert r['status']==receipt['status']=='complete' and receipt['constructor_calls']==6
    c=m['runner_config']
    for p,h in c['resources'].items():assert sha(p)==h,p
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(RUN/c['runtime_tool']),'--verify',linux_path(c['linux_identity'])],check=True)
    threads=set();identities={};records=[]
    for row in r['trials']:
        folder=out/f"{row['draw']}_{row['arm']}"
        answer=(folder/'answer.md').read_text(encoding='utf-8')
        events=[json.loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        collection=audit_events(events,answer);assert collection['thread_id'] not in threads;threads.add(collection['thread_id'])
        assert collection['usage']==row['collection']['usage'] and collection['tool_calls']==0
        prompt=Path(c['prompts'][row['arm']]).read_text(encoding='utf-8').encode('utf-8')
        assert hashlib.sha256(prompt).hexdigest()==row['collection']['prompt_transport']['sha256']
        assert len(prompt)==row['collection']['prompt_transport']['bytes_written']
        source=(folder/('proposal.py' if row['arm']=='python' else 'proposal.wspl')).read_text(encoding='utf-8')
        assert source==extract(answer,row['arm'])
        for name,expected_hash in row['candidate_hashes'].items():
            candidate_path=folder/(name+'_candidate.npz');assert sha(candidate_path)==expected_hash
            with np.load(folder/(name+'_parent.npz')) as p,np.load(candidate_path) as a,np.load(Path(m['constructor_kwargs']['reference_root'])/(name+'.npz')) as original:
                for k in p.files:assert p[k].tobytes()==original[k].tobytes()
                expected={k:p[k].copy() for k in p.files}
                # Independent inverse-index construction using image-axis roll.
                ix=np.roll(np.arange(144).reshape(12,12),2,axis=1).ravel()
                expected['W1'][:144]=p['W1'][ix];expected['W3']=p['W3'][:,ix];expected['b3']=p['b3'][ix]
                for k in a.files:assert expected[k].tobytes()==a[k].tobytes()
                identity=hashlib.sha256(b''.join(a[k].tobytes() for k in sorted(a.files))).hexdigest()
                if name in identities:assert identities[name]==identity
                identities[name]=identity
            with np.load(folder/(name+'_paired_samples.npz')) as z:
                assert set(z.files)=={'parent','candidate','target'}
                for k in z.files:assert z[k].shape==(64,144) and z[k].dtype==np.float32 and np.isfinite(z[k]).all()
                target=np.roll(z['parent'].reshape(64,12,12),2,axis=2).reshape(64,144)
                assert np.array_equal(target,z['target'])
                error=float(np.abs(z['candidate']-target).max())
                assert error==row['evaluation'][name]['sample_max_abs_error'] and error<=.001
        records.append(dict(arm=row['arm'],draw=row['draw'],usage=collection['usage'],source_bytes=len(source.encode()),
            construction_seconds=row['construction_seconds'],source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            syntax='bounded loops and immutable parent copies' if row['arm']=='wspl' else 'NumPy index permutation'))
    assert len(records)==6 and len(threads)==6
    summary={}
    for arm in ('python','wspl'):
        rows=[x for x in records if x['arm']==arm]
        summary[arm]=dict(successes=3,certificates=9,usage={k:sum(x['usage'][k] for x in rows) for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')},construction_seconds=sum(x['construction_seconds'] for x in rows))
    result=dict(status='passed',sources=len(m['files']),resources=len(c['resources']),fresh_threads=6,
        candidates=18,paired_samples=1152,all_arms_bitwise_identical_per_parent=True,summary=summary,records=records,
        active_seconds=r['active_seconds'],finished_utc=receipt['finished_utc'],scope='No new model calls, construction execution or sampled rollouts; source/resource/event/array/archive audit.')
    (MAIN/'docs/wspl/diffusion_edit_v1_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
