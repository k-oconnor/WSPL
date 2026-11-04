"""Read-only completion audit against the frozen Breakout source and artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PIN='e842840acc4c8229f370e37c456bf6898f88f676a13e3dde36b2b3ba5085acc4'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def run(root):
    root=root.resolve();sys.path[:0]=[str(root/'src'),str(root/'experiments')]
    import numpy as np
    from source_consistency import verify
    from skillforge.csail_breakout import dependency_identity,BreakoutPolicy,encode_observation,probabilities,sample_action
    from skillforge.codex_one_shot import audit_events,audit_preview,command
    from skillforge.wsp_breakout import ReworkedBreakoutPolicy,prompt,frozen_checks,stored_equations
    from skillforge.wsp_artifact import TensorArtifact
    from skillforge.wsp_editor import EditPlan,apply
    from run_wsp_breakout import ORDER,comparisons
    out=root/'results/wsp_breakout_v1';mp=root/'experiments/source_pins/wsp_breakout_v1.manifest.json'
    m=verify(root,mp,PIN);c=m['runner_config'];r=read(out/'results.json');receipt=read(out/'source_receipt.json')
    assert r==read(out/'status.json') and r['status']==receipt['status']=='complete'
    assert receipt['constructor_calls']==r['planned_invocations']==r['attempted_invocations']==6
    assert receipt['manifest_sha256']==r['manifest_sha256']==sha(out/'approved_manifest.json')==PIN
    for p,h in c['bound_files'].items():assert sha(root/p)==h,p
    assert dependency_identity()['sha256']==c['dependencies_sha256']
    cli=read(root/c['local_paths'])['codex_cli'];assert sha(Path(cli))==c['cli_sha256']
    assert [(t['draw'],t['arm']) for t in r['trials']]==list(ORDER)
    assert r['comparisons']==comparisons(r['trials'])
    guide=(root/c['guide']).read_text(encoding='utf-8');reference=(root/c['reference_prompt']).read_text(encoding='utf-8')
    threads=set();rows=[];hashes={};traced_steps=0
    for t in r['trials']:
        assert t['status']=='complete' and t['completed_episodes']==1000
        folder=out/f"draw_{t['draw']}_{t['arm']}"
        for name,h in t['artifact_files'].items():assert sha(folder/name)==h,name
        assert read(folder/'artifact_index.json')==t['artifact_files']
        answer=(folder/'answer.md').read_text(encoding='utf-8')
        events=[json.loads(x) for x in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        audit=audit_events(events,answer);saved=read(folder/'collection.json')
        assert saved==t['collection'] and audit['agent_messages']==1 and audit['tool_calls']==0
        for k,v in audit.items():assert saved[k]==v
        assert audit['thread_id'] not in threads;threads.add(audit['thread_id'])
        text=(folder/'prompt.txt').read_text(encoding='utf-8')
        assert text==prompt(reference,guide,t['arm'])==(root/c['prompts'][t['arm']]).read_text(encoding='utf-8')
        audit_preview(read(folder/'prompt_preview.json'),text)
        assert read(folder/'command.json')==command(cli,root/c['catalog'],root/c['instructions'],(folder/'empty').resolve(),(folder/'answer.md').resolve())
        policy=ReworkedBreakoutPolicy();artifact,plan,construction=policy.construct(t['arm'],answer)
        assert artifact.sha256==t['candidate_id']==TensorArtifact.load(folder/'candidate.json').sha256
        assert plan.sha256==EditPlan.load(folder/'edit.plan.json').sha256
        assert apply(plan,parent=policy.parent,expected_parent_id=policy.parent.sha256).sha256==artifact.sha256
        assert construction==read(folder/'construction.json')
        assert plan.to_dict()['source']==(folder/'source.wspl').read_text(encoding='utf-8')
        deployed=BreakoutPolicy(folder/'policy.npz',sha(folder/'policy.npz'))
        assert policy.tensor_identity()==deployed.tensor_identity()==construction['tensor_sha256']
        assert frozen_checks(deployed,artifact)==read(folder/'deployment_check.json')
        assert stored_equations(artifact)==read(folder/'stored_equations.json')
        records=[json.loads(x) for x in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records)==1000
        for i,e in enumerate(records):
            assert (e['index'],e['draw'],e['arm'],e['env_seed'],e['action_seed'])==(i,t['draw'],t['arm'],1110000+i,1210000+i)
            assert e['status']=='complete' and 1<=e['steps']<=10000
            assert len(e['action_counts'])==6 and sum(e['action_counts'])==e['steps']
            assert all(type(n) is int and n>=0 for n in e['action_counts'])
            assert e['truncated']==(not e['terminal']) and (e['terminal'] or e['steps']==10000)
            assert np.isfinite(e['reward']) and 0<=e['reward']<=e['steps']
        values=np.array([e['reward'] for e in records],dtype=np.float64)
        summary=dict(episodes=1000,mean=float(values.mean()),se=float(values.std(ddof=1)/np.sqrt(1000)),
            min=float(values.min()),max=float(values.max()),mean_length=float(np.mean([e['steps'] for e in records])),
            truncations=sum(e['truncated'] for e in records))
        assert summary==t['summary']
        for i in range(3):
            e=records[i]
            with np.load(folder/f'trajectory_{i}.npz',allow_pickle=False) as trace:
                assert set(trace.files)=={'states','requested_actions','rewards','terminals'}
                states=trace['states'];actions=trace['requested_actions'];rewards=trace['rewards'];terminals=trace['terminals']
                assert states.dtype==np.bool_ and states.shape==(e['steps'],10,10,4)
                assert all(len(x)==e['steps'] for x in (actions,rewards,terminals))
                assert rewards.sum()==e['reward'] and bool(terminals[-1])==e['terminal'] and not terminals[:-1].any()
                assert np.bincount(actions,minlength=6).tolist()==e['action_counts']
                rng=np.random.Generator(np.random.PCG64(e['action_seed']))
                for state,action in zip(states,actions):
                    assert sample_action(probabilities(deployed.logits(encode_observation(state))),rng.random())==int(action)
                traced_steps+=len(actions)
        rows.append(dict(draw=t['draw'],arm=t['arm'],candidate_id=artifact.sha256,summary=summary,
            numeric_profile=t['numeric_profile'],nonzero=t['nonzero'],resources=t['resources'],
            source_bytes=t['source_bytes'],reply_bytes=(folder/'answer.md').stat().st_size,
            rounding_changes=plan.to_dict()['rounding']['changed'],usage=audit['usage'],
            collection_seconds=t['collection']['active_seconds'],construction_seconds=t['construction_seconds'],
            evaluation_seconds=t['evaluation_seconds']))
        hashes.update({folder.name+'/'+p:h for p,h in t['artifact_files'].items() if not p.startswith('offline-home/')})
    arms={}
    for arm in ('native','wspl'):
        ts=[t for t in rows if t['arm']==arm]
        means=[t['summary']['mean'] for t in ts]
        arms[arm]=dict(proposals=3,valid=3,mean=float(np.mean(means)),across_proposal_sd=float(np.std(means,ddof=1)),
            per_proposal_means=means,source_bytes=sum(t['source_bytes'] for t in ts),
            reply_bytes=sum(t['reply_bytes'] for t in ts),
            usage={k:sum(t['usage'].get(k,0) for t in ts) for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')},
            collection_seconds=sum(t['collection_seconds'] for t in ts),construction_seconds=sum(t['construction_seconds'] for t in ts),
            evaluation_seconds=sum(t['evaluation_seconds'] for t in ts),truncations=sum(t['summary']['truncations'] for t in ts))
    for name in ('results.json','status.json','source_receipt.json','approved_manifest.json','dependencies.json'):hashes[name]=sha(out/name)
    verify(root,mp,PIN)
    return dict(status='passed',manifest_sha256=PIN,source_files=len(m['files']),bound_files=len(c['bound_files']),source_unchanged=True,
        dependencies_and_cli_verified=True,unique_threads=len(threads),tool_calls=0,reply_candidate_plan_replays=6,
        episode_records_checked=6000,trajectories_checked=18,recorded_actions_replayed=traced_steps,
        trajectory_check='frozen recorded observations and action RNG; no simulator episodes rerun',
        started_utc=receipt['started_utc'],finished_utc=receipt['finished_utc'],active_seconds=r['active_seconds'],
        decision=r['comparisons']['decision'],arms=arms,trials=rows,raw_sha256=hashes,
        auditor_sha256=sha(Path(__file__)),new_model_calls=0,new_scored_episodes=0,training_updates=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.root)
    with args.output.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials','raw_sha256')},indent=2))
