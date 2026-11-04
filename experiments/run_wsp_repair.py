"""Source-gated two-interface repair screen. No direct scientific entry point."""
import json
from pathlib import Path
import time

from skillforge.wsp_repair import TASKS,ARMS,public_task,prompt
from skillforge.wsp_artifact import digest
from skillforge.codex_one_shot import collect
from skillforge.csail_breakout import file_hash
from skillforge.breakout_matched import matched_dependency_identity
from run_csail_breakout import atomic_json


ORDER=tuple((t,draw,arm) for draw in range(2) for i,t in enumerate(TASKS)
            for arm in (ARMS if (draw+i)%2==0 else ARMS[::-1]))


def comparisons(trials):
    arms={a:dict(planned=6,successes=sum(t.get('evaluation',{}).get('success',False) for t in trials if t['arm']==a),
        valid=sum(t['status']=='evaluated' for t in trials if t['arm']==a),
        invalid=sum(t['status']=='invalid_proposal' for t in trials if t['arm']==a),
        usage={k:sum(t.get('collection',{}).get('usage',{}).get(k,0) for t in trials if t['arm']==a)
            for k in ('input_tokens','cached_input_tokens','output_tokens')}) for a in ARMS}
    complete=len(trials)==12 and all(t['status'] in ('evaluated','invalid_proposal') for t in trials)
    paired=[]
    for task,draw in [(t,d) for d in range(2) for t in TASKS]:
        rows={r['arm']:r for r in trials if r['task']==task and r['draw']==draw}
        resolved=len(rows)==2 and all(r['status'] in ('evaluated','invalid_proposal') for r in rows.values())
        paired.append(dict(task=task,draw=draw,resolved=resolved,
            wspl_minus_direct=(int(rows['wspl'].get('evaluation',{}).get('success',False))-int(rows['direct'].get('evaluation',{}).get('success',False))) if resolved else None))
    decision='unresolved_incomplete'
    if complete:
        if arms['direct']['successes']<4:decision='baseline_feasibility_unresolved'
        elif arms['wspl']['successes']<5 or arms['wspl']['successes']<arms['direct']['successes']:decision='no_go_for_this_configuration'
        elif arms['direct']['successes']==arms['wspl']['successes']==6:decision='feasible_but_ceiling_no_advantage_established'
        else:decision='feasible_for_larger_repair_study'
    return dict(arms=arms,paired_success_differences=paired,complete=complete,decision=decision,
                scope='descriptive feasibility; six task/draw pairs, three task families; no superiority or human-time claim')


def execute_trials(constructor,out,config,collector,stop,snapshot):
    trials=[dict(task=t,draw=d,arm=a,status='not_attempted') for t,d,a in ORDER];threads=set()
    guide=Path(config['guide']).read_text(encoding='utf-8')
    state=dict(status='running',trials=trials)
    snapshot(state)
    for t in trials:
        if stop():state['status']='paused';break
        study=constructor();folder=out/f'{t["task"]}_{t["draw"]}_{t["arm"]}'
        t['status']='collecting';snapshot(state)
        text=Path(config['prompts'][t['task']][t['arm']]).read_text(encoding='utf-8')
        if text!=prompt(t['task'],t['arm'],guide):raise ValueError('frozen prompt differs from current task/guide')
        try:
            answer,audit=collector(folder,text,stop)
            t['collection']=audit
            if audit['thread_id'] in threads:raise ValueError('collection reused a thread')
            threads.add(audit['thread_id'])
        except Exception as exc:
            t.update(status='collection_failed',error=f'{type(exc).__name__}: {exc}');state['status']='incomplete';snapshot(state);break
        started=time.perf_counter();t['status']='constructing';snapshot(state)
        try:
            candidate,receipt=study.construct(t['task'],t['arm'],answer)
        except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,SyntaxError,RecursionError) as exc:
            t.update(status='invalid_proposal',error=f'{type(exc).__name__}: {exc}',construction_seconds=time.perf_counter()-started)
            snapshot(state);continue
        t.update(construction_seconds=time.perf_counter()-started,candidate_id=candidate.sha256)
        candidate.save(folder/'candidate.json');atomic_json(folder/'construction.json',receipt)
        started=time.perf_counter();evaluation=study.evaluate(t['task'],candidate)
        atomic_json(folder/'evaluation.json',evaluation)
        t.update(status='evaluated',evaluation_seconds=time.perf_counter()-started,evaluation=evaluation,
                 resources=receipt['resources'],structured_operations=[r['operation'] for r in receipt.get('wspl_plan',{}).get('graph_edits',{}).get('operations',[])] if receipt.get('wspl_plan',{}).get('graph_edits') else [])
        snapshot(state)
    if state['status']=='running':state['status']='complete'
    state.update(comparisons=comparisons(trials),planned_invocations=12,
                 attempted_invocations=sum(t['status']!='not_attempted' for t in trials),training_updates=0,local_gpu_used=False)
    snapshot(state);return state


def run(constructor,output_dir,config):
    out=Path(output_dir).resolve();receipt=json.loads((out/'source_receipt.json').read_text(encoding='utf-8'))
    if receipt['status']!='verified' or file_hash(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('approved source receipt required')
    if (config['call_guard_seconds'],config['wall_guard_seconds'])!=(1200,14400):raise ValueError('frozen usage guards required')
    def verify_inputs():
        for p,h in config['bound_files'].items():
            if file_hash(p)!=h:raise ValueError('bound input changed: '+p)
        if matched_dependency_identity()['sha256']!=config['dependencies_sha256']:raise ValueError('dependency identity differs')
        if config['task_ids']!={t:digest(public_task(t)) for t in TASKS}:raise ValueError('task identities differ')
    verify_inputs();cli=json.loads(Path(config['local_paths']).read_text())['codex_cli']
    if file_hash(cli)!=config['cli_sha256']:raise ValueError('CLI identity differs')
    atomic_json(out/'dependencies.json',matched_dependency_identity())
    started=time.monotonic();last={}
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def snapshot(state):
        state.update(active_seconds=time.monotonic()-started,manifest_sha256=receipt['manifest_sha256']);last.update(state);atomic_json(out/'status.json',state)
    def collector(folder,text,should_stop):
        return collect(cli,config['catalog'],config['instructions'],folder,text,config['call_guard_seconds'],should_stop)
    try:
        final=execute_trials(constructor,out,config,collector,stop,snapshot)
        verify_inputs();atomic_json(out/'results.json',final)
    except BaseException as exc:
        last.update(status='failed',error=f'{type(exc).__name__}: {exc}');snapshot(last);atomic_json(out/'results.json',last);raise


if __name__=='__main__':raise SystemExit('Use source_consistency.py with an exact user-approved source pin.')
