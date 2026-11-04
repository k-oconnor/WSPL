"""Approved-source three-interface ReLU authoring screen; no direct execution."""
import json
import os
from pathlib import Path
import statistics
import time

from skillforge.breakout_matched import NODE_LIMIT
from skillforge.breakout_matched import matched_dependency_identity as dependency_identity
from skillforge.codex_one_shot import collect
from skillforge.csail_breakout import episode,file_hash,summarize
from skillforge.wspl_equiv import check_equivalence
from run_csail_breakout import atomic_json

ORDER=(('weights',0),('code',0),('wspl',0),('code',1),('wspl',1),('weights',1),('wspl',2),('weights',2),('code',2))


def contrasts(trials):
    """Predeclared zero penalty for invalid proposals; incomplete remains unresolved."""
    arms={}
    for arm in ('weights','code','wspl'):
        rows=[t for t in trials if t['arm']==arm]
        scores=[t['summary']['mean'] if t['status']=='complete' else 0. for t in rows]
        valid=[t['summary']['mean'] for t in rows if t['status']=='complete']
        arms[arm]=dict(planned=3,evaluated=sum(t['status']=='complete' for t in rows),
            invalid=sum(t['status']=='invalid_proposal' for t in rows),
            failure_adjusted_mean=sum(scores)/3,policy_scores=scores,
            valid_only_mean=statistics.mean(valid) if valid else None,
            valid_policy_sample_sd=statistics.stdev(valid) if len(valid)>1 else None,
            reported_usage={k:sum(t.get('collection',{}).get('usage',{}).get(k,0) for t in rows)
                for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')})
    all_resolved=all(t['status'] in ('complete','invalid_proposal') for t in trials)
    baseline_useful=all(arms[a]['failure_adjusted_mean']>=100 for a in ('weights','code'))
    wspl_useful=arms['wspl']['failure_adjusted_mean']>=100
    ratios={a:arms['wspl']['failure_adjusted_mean']/arms[a]['failure_adjusted_mean'] if arms[a]['failure_adjusted_mean'] else None for a in ('weights','code')}
    gate='unresolved'
    if all_resolved:
        if not baseline_useful:gate='baseline_feasibility_unresolved'
        else:gate='go' if wspl_useful and arms['wspl']['evaluated']==3 and all(v>=.9 for v in ratios.values()) else 'no_go_for_this_configuration'
    return dict(arms=arms,wspl_to_baseline_ratios=ratios,internal_feasibility_decision=gate,
        scope='descriptive pilot decision, not confirmed noninferiority, assurance or commercial benefit')


def run(constructor,output_dir,config):
    out=Path(output_dir).resolve();receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or file_hash(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('approved-source receipt required')
    def verify_inputs():
        for p,h in config['bound_files'].items():
            if file_hash(p)!=h:raise ValueError('bound file changed: '+p)
        if dependency_identity()['sha256']!=config['dependencies_sha256']:raise ValueError('core dependencies differ')
    verify_inputs()
    atomic_json(out/'dependencies.json',dependency_identity())
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(k)!='1':raise ValueError(k+' must equal 1 at startup')
    if (config['episodes'],config['max_steps'],config['call_guard_seconds'],config['wall_guard_seconds'])!=(1000,10000,1200,14400):raise ValueError('frozen screen limits required')
    cli=json.loads(Path(config['local_paths']).read_text())['codex_cli']
    if file_hash(cli)!=config['cli_sha256']:raise ValueError('CLI identity differs')
    import numpy as np
    from minatar import Environment
    started=time.monotonic();threads=set()
    trials=[dict(arm=a,draw=i,status='not_attempted',summary=None) for a,i in ORDER]
    state=dict(status='running',trials=trials,source_manifest_sha256=receipt['manifest_sha256'])
    def snapshot():
        state['active_seconds']=time.monotonic()-started
        atomic_json(out/'status.json',state)
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>14400 or (p.exists() and json.loads(p.read_text()).get('stop'))
    try:
        for t in trials:
            if stop():state['status']='paused';break
            arm=t['arm'];folder=out/f'{arm}_{t["draw"]}';policy=constructor()
            t['status']='collecting';snapshot()
            prompt=Path(config['prompts'][arm]).read_text(encoding='utf-8')
            try:
                answer,audit=collect(cli,config['catalog'],config['instructions'],folder,prompt,1200,stop)
                if audit['thread_id'] in threads:raise ValueError('thread reused')
                threads.add(audit['thread_id']);t['collection']=audit
            except Exception as exc:
                t.update(status='collection_failed',error=f'{type(exc).__name__}: {exc}');state['status']='incomplete';break
            t['status']='compiling';snapshot();began=time.monotonic()
            try:t['construction']=policy.load_answer(arm,answer)
            except (ValueError,TypeError,SyntaxError,ArithmeticError,RecursionError) as exc:
                t.update(status='invalid_proposal',error=f'{type(exc).__name__}: {exc}',construction_seconds=time.monotonic()-began)
                snapshot();continue
            t['construction_seconds']=time.monotonic()-began
            np.savez(folder/'policy.npz',**policy.weights);t['weights_sha256']=file_hash(folder/'policy.npz')
            policy.artifact.save(folder/'deployment.relu.json')
            if policy.program is not None:
                atomic_json(folder/'source.ir.json',policy.program)
                if policy.expanded_source is not None:(folder/'expanded.wspl').write_text(policy.expanded_source,encoding='utf-8')
                t['source_equivalence']=check_equivalence(policy.program,policy.artifact,timeout_ms=5000,max_parameters=20486,max_source_nodes=NODE_LIMIT)
                atomic_json(folder/'source_equivalence.json',t['source_equivalence'])
            else:t['source_equivalence']=dict(status='NOT_APPLICABLE',reason='native weights have no independent scalar source')
            t['status']='evaluating';snapshot();rows=[];last=time.monotonic();began=time.monotonic()
            for i in range(1000):
                if stop():state['status']='paused';break
                env=Environment('breakout',sticky_action_prob=.1,difficulty_ramping=True)
                if env.env.channels!={'paddle':0,'ball':1,'trail':2,'brick':3}:raise ValueError('channel schema differs')
                r=episode(policy,env,env_seed=710000+i,action_seed=810000+i,max_steps=10000,should_stop=stop)
                if r['status']!='complete':t['discarded_partial_episode']=r;state['status']='paused';break
                r.update(arm=arm,draw=t['draw'],index=i)
                with (out/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
                rows.append(r);t['completed_episodes']=len(rows)
                if time.monotonic()-last>30:snapshot();last=time.monotonic()
            if policy.tensor_identity()!=t['construction']['tensor_sha256']:raise ValueError('frozen tensors changed')
            t['evaluation_seconds']=time.monotonic()-began
            t['summary']=summarize(rows) if rows else None
            t['status']='complete' if len(rows)==1000 else 'evaluation_incomplete'
            if state['status']=='paused':break
        if state['status']=='running':state['status']='complete'
        verify_inputs()
    except BaseException:
        state['status']='failed';raise
    finally:
        state.update(planned_invocations=9,attempted_invocations=sum(t['status']!='not_attempted' for t in trials),
            comparisons=contrasts(trials),training_updates=0,local_gpu_used=False,
            scope='matched three-interface fixed ReLU authoring pilot; not native tanh replication or robot/formal game proof')
        snapshot();atomic_json(out/'results.json',state)


if __name__=='__main__':raise SystemExit('Use source_consistency.py with an exact user-approved manifest.')
