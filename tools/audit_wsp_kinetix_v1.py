"""Read-only two-runtime audit; no model calls, fitting or simulator steps."""
import json
import os
from pathlib import Path
import sys
import hashlib
import numpy as np

BASE=Path('C:/Users/kevin/OneDrive/Desktop') if os.name=='nt' else Path('/mnt/c/Users/kevin/OneDrive/Desktop')
ROOT=BASE/'Skillforge-wsp-kinetix-v1'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify,digest
from skillforge.wsp_kinetix import KinetixRepair,prompt
from skillforge.kinetix_replay import sample_action
from skillforge.codex_one_shot import audit_events,audit_preview
from run_wsp_kinetix import comparison

PIN='0cd0a7a6f7e7e0b6c5a8e2348788dc127f3d435a187c048c35bc56ed760f23e3'


def main():
    os.chdir(ROOT);out=ROOT/'results/wsp_kinetix_v1'
    m=verify(ROOT,out/'approved_manifest.json',PIN);c=m['runner_config']
    receipt=json.loads((out/'source_receipt.json').read_text())
    result=json.loads((out/'results.json').read_text())
    assert receipt['status']==result['status']=='complete' and receipt['constructor_calls']==7
    assert receipt['manifest_sha256']==result['manifest_sha256']==PIN
    resources=c['bound_files'] if os.name=='nt' else c['linux_resources']
    for path,h in resources.items():assert digest(Path(path))==h,path
    assert len(result['trials'])==6 and comparison(result['trials'])==result['comparison']
    counts=0;actions=0;threads=set();arm_stats={}
    for name in ['parent_control']+[f"draw_{t['draw']}_{t['arm']}" for t in result['trials']]:
        folder=out/name;rows=[json.loads(x) for x in (folder/'episodes.jsonl').read_text().splitlines()]
        assert len(rows)==1024 and [r['index'] for r in rows]==list(range(1024))
        for r in rows:assert r['seed']==10100000+r['index'] and 1<=r['steps']<=256 and np.isfinite(r['return_'])
        summary=json.loads((folder/'evaluation.json').read_text())
        assert summary['outcomes']=={o:sum(r['outcome']==o for r in rows) for o in ('success','red','timeout','invalid')}
        assert summary['success_rate']==summary['outcomes']['success']/1024
        assert summary['mean_length']==float(np.mean([r['steps'] for r in rows]))
        assert summary['mean_return']==float(np.mean([r['return_'] for r in rows]))
        counts+=len(rows)
        if os.name!='nt':
            policy=object.__new__(KinetixRepair)
            with np.load(folder/'policy.npz',allow_pickle=False) as z:policy.weights=dict(z)
            for i in range(3):
                trace=json.loads((folder/f'trajectory_{i}.json').read_text());rng=np.random.default_rng(rows[i]['seed'])
                assert len(trace)==rows[i]['steps']
                assert abs(sum(t['reward'] for t in trace)-rows[i]['return_'])<1e-12
                for j,t in enumerate(trace):
                    assert t['step']==j and t['done']==(j==len(trace)-1)
                    assert np.array_equal(sample_action(policy.logits(np.asarray(t['observation'],np.float32)),rng),t['action'])
                    actions+=1
    if os.name=='nt':
        parent=KinetixRepair(**m['constructor_kwargs'])
        with np.load(out/'parent_control/policy.npz',allow_pickle=False) as z:
            for k,a in parent.original.items():assert np.array_equal(z[k].view('u4'),a.view('u4'))
        for t in result['trials']:
            folder=out/f"draw_{t['draw']}_{t['arm']}"
            policy=KinetixRepair(**m['constructor_kwargs'])
            text=(folder/'prompt.txt').read_text(encoding='utf-8')
            assert text==prompt(Path(c['reference_prompt']).read_text(encoding='utf-8'),Path(c['guide']).read_text(encoding='utf-8'),t['arm'],policy.parent.sha256)
            answer=(folder/'answer.md').read_text(encoding='utf-8')
            events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            audit=audit_events(events,answer);assert audit['usage']==t['collection']['usage']
            assert audit['thread_id'] not in threads;threads.add(audit['thread_id'])
            h=hashlib.sha256(text.encode()).hexdigest()
            assert t['collection']['prompt_transport']['sha256']==h
            assert t['collection']['prompt_transport']['bytes_written']==len(text.encode())
            audit_preview(json.loads((folder/'prompt_preview.json').read_text(encoding='utf-8')),'Full prompt supplied through exec stdin; SHA-256 '+h)
            artifact,plan,source=policy.construct(t['arm'],answer)
            assert artifact.sha256==t['candidate_id']
            assert artifact.to_dict()==json.loads((folder/'candidate.json').read_text())
            saved_plan=json.loads((folder/'edit.plan.json').read_text())
            assert plan.to_dict()==saved_plan['body'] and plan.sha256==saved_plan['plan_id']
            assert source==(folder/'source.wspl').read_text(encoding='utf-8')
            assert len(source.encode())==t['source_bytes']
            assert sum(int(np.count_nonzero(policy.weights[k]!=policy.original[k])) for k in policy.weights)==t['changed_parameters']
            with np.load(folder/'policy.npz',allow_pickle=False) as z:
                for k,a in policy.weights.items():assert np.array_equal(z[k].view('u4'),a.view('u4'))
        for arm in ('native','wspl'):
            ts=[t for t in result['trials'] if t['arm']==arm]
            arm_stats[arm]={k:sum(t['collection']['usage'][k] for t in ts) for k in ('input_tokens','output_tokens','reasoning_output_tokens','cached_input_tokens')}
            arm_stats[arm].update(source_bytes=sum(t['source_bytes'] for t in ts),collection_seconds=sum(t['collection']['active_seconds'] for t in ts),changed_parameters=[t['changed_parameters'] for t in ts],mean_steps=float(np.mean([t['evaluation']['mean_length'] for t in ts])))
    phase='windows' if os.name=='nt' else 'linux'
    report=dict(status='audit_passed',phase=phase,source_files=len(m['files']),bound_files=len(resources),manifest_sha256=PIN,
        episodes=counts,recorded_actions_replayed=actions,arm_stats=arm_stats,active_seconds=result['active_seconds'],finished_utc=receipt['finished_utc'],
        auditor_sha256=digest(Path(__file__)),scope='Read-only source/resource, record and artifact audits; no new physics steps. Windows reconstructs edits/collection; Linux replays saved actions.')
    dest=BASE/'Skillforge/docs/wspl'/f'wsp_kinetix_v1_audit_{phase}.json';dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
