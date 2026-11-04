"""Read-only completion audit; no simulator steps or policy fitting."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/c/Users/kevin/OneDrive/Desktop/Skillforge-kinetix-replay-v1')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify
from run_kinetix_replay import verify_resources
from skillforge.kinetix_replay import ReplayStudy,sample_action,tensor_identity,sha


def main():
    import os
    os.chdir(ROOT)
    digest='ef3ca419ffae13e156be92a3eb2d220762c035cc2eb00839e4d8c3c8d0b0046a'
    manifest=verify(ROOT,ROOT/'experiments/source_pins/kinetix_replay_v1.manifest.json',digest)
    verify_resources(manifest['runner_config'])
    out=ROOT/'results/kinetix_replay_v1'
    receipt=json.loads((out/'source_receipt.json').read_text())
    result=json.loads((out/'results.json').read_text())
    assert receipt['status']==result['status']=='complete'
    assert receipt['constructor_calls']==1
    assert receipt['manifest_sha256']==result['manifest_sha256']==sha(out/'approved_manifest.json')==digest
    study=ReplayStudy(**manifest['constructor_kwargs'])
    rows=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines()]
    assert len(rows)==2048
    summaries={};traces=0;actions=0
    for mode,policy in study.policies.items():
        with np.load(out/(mode+'_actor.npz'),allow_pickle=False) as z:
            assert tensor_identity(dict(z))==policy.tensor_sha256
            for key,a in policy.weights.items():
                assert np.array_equal(z[key].view('u8'),a.view('u8'))
        selected=[r for r in rows if r['mode']==mode]
        assert len(selected)==1024
        assert [r['episode'] for r in selected]==list(range(1024))
        for r in selected:
            assert r['seed']==9100000+r['episode']
            assert r['policy_sha256']==policy.tensor_sha256
            assert 1<=r['steps']<=256 and np.isfinite(r['return_'])
            assert r['outcome'] in ('success','red','timeout','invalid')
        summary=dict(episodes=len(selected),outcomes={k:sum(r['outcome']==k for r in selected) for k in ('success','red','timeout','invalid')},
            mean_return=float(np.mean([r['return_'] for r in selected])),mean_length=float(np.mean([r['steps'] for r in selected])))
        assert summary==result['summaries'][mode]
        summaries[mode]=summary
        for episode in range(3):
            trace=json.loads((out/f'{mode}_{episode}_trajectory.json').read_text())
            row=selected[episode];rng=np.random.default_rng(row['seed'])
            assert len(trace)==row['steps']
            assert abs(sum(t['reward'] for t in trace)-row['return_'])<1e-12
            for i,t in enumerate(trace):
                assert t['step']==i
                assert np.array_equal(sample_action(policy.logits(np.asarray(t['observation'],dtype=np.float32)),rng),t['action'])
                assert t['done']==(i==len(trace)-1)
                actions+=1
            traces+=1
    assert [(r['episode'],r['mode']) for r in rows]==[(i,m) for i in range(1024) for m in ('parent','edited')]
    report=dict(status='audit_passed',manifest_sha256=digest,sources=len(manifest['files']),resources=len(manifest['runner_config']['resources']),
        source_receipt_status=receipt['status'],constructor_calls=receipt['constructor_calls'],finished_utc=receipt['finished_utc'],
        episodes=len(rows),trajectories=traces,recorded_actions_replayed=actions,active_seconds=result['active_seconds'],
        summaries=summaries,reconstruction=study.reconstruction,auditor_sha256=sha(Path(__file__)),
        scope='Source/runtime and artifact identities, episode accounting, stored-action RNG replay and recorded return sums; zero new simulator steps. Full physics trajectories not independently rerun.')
    destination=Path('/mnt/c/Users/kevin/OneDrive/Desktop/Skillforge/docs/wspl/kinetix_replay_v1_results.json')
    destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
