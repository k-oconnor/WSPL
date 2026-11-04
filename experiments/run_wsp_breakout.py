"""Source-gated original-tanh Breakout screen with complete candidate archives."""
import json
import os
from pathlib import Path
import time

import numpy as np

from skillforge.wsp_breakout import ARMS, prompt, frozen_checks, stored_equations
from skillforge.wsp_artifact import TensorArtifact
from skillforge.codex_one_shot import collect
from skillforge.csail_breakout import BreakoutPolicy, dependency_identity, episode, file_hash, summarize
from run_csail_breakout import atomic_json

ORDER = ((0,'native'),(0,'wspl'),(1,'wspl'),(1,'native'),(2,'native'),(2,'wspl'))
RESOLVED = ('complete','invalid_proposal')


def index_artifacts(folder):
    if not folder.exists():
        return {}
    files={p.relative_to(folder).as_posix():file_hash(p) for p in sorted(folder.rglob('*'))
           if p.is_file() and p.name!='artifact_index.json'}
    atomic_json(folder/'artifact_index.json',files)
    return files


class TraceEnvironment:
    """Record the first three primary episodes without adding simulator calls."""
    def __init__(self, env):
        self.wrapped = env
        self.states = []; self.actions = []; self.rewards = []; self.terminals = []

    def seed(self, seed):
        return self.wrapped.seed(seed)

    def reset(self):
        return self.wrapped.reset()

    def state(self):
        state = self.wrapped.state()
        self.states.append(state.copy())
        return state

    def act(self, action):
        reward, terminal = self.wrapped.act(action)
        self.actions.append(action); self.rewards.append(reward); self.terminals.append(terminal)
        return reward, terminal

    def save(self, path):
        if len(self.states) != len(self.actions):
            raise ValueError('trace observation/action accounting differs')
        np.savez_compressed(path, states=np.asarray(self.states,dtype=np.bool_),
                            requested_actions=np.asarray(self.actions,dtype=np.uint8),
                            rewards=np.asarray(self.rewards,dtype=np.float32),
                            terminals=np.asarray(self.terminals,dtype=np.bool_))


def comparisons(trials):
    complete = all(t['status'] in RESOLVED for t in trials) and len(trials)==6
    arms = {}
    for arm in ARMS:
        rows = [t for t in trials if t['arm']==arm]
        values = [t['summary']['mean'] if t['status']=='complete' else 0.
                  for t in rows if t['status'] in RESOLVED]
        arms[arm] = dict(planned=3, evaluated=sum(t['status']=='complete' for t in rows),
            invalid=sum(t['status']=='invalid_proposal' for t in rows),
            failure_adjusted_mean=float(np.mean(values)) if len(values)==3 else None,
            per_proposal_returns=values,
            valid_only_mean=(float(np.mean([t['summary']['mean'] for t in rows if t['status']=='complete']))
                             if any(t['status']=='complete' for t in rows) else None),
            across_proposal_sd=float(np.std(values,ddof=1)) if len(values)==3 else None,
            usage={k:sum(t.get('collection',{}).get('usage',{}).get(k,0) for t in rows)
                   for k in ('input_tokens','cached_input_tokens','output_tokens')})
    decision = 'unresolved_incomplete'
    if complete:
        native, wspl = arms['native'], arms['wspl']
        if native['failure_adjusted_mean'] < 100:
            decision = 'baseline_feasibility_unresolved'
        elif wspl['evaluated']==3 and wspl['failure_adjusted_mean']>=100 and wspl['failure_adjusted_mean']>=.9*native['failure_adjusted_mean']:
            decision = 'feasible_for_next_domain'
        else:
            decision = 'no_go_for_this_configuration'
    paired = []
    for draw in range(3):
        rows = {t['arm']:t for t in trials if t['draw']==draw}
        resolved = len(rows)==2 and all(t['status'] in RESOLVED for t in rows.values())
        scores = {a:(t['summary']['mean'] if t['status']=='complete' else 0.) for a,t in rows.items()}
        paired.append(dict(draw=draw, resolved=resolved,
                           wspl_minus_native=scores['wspl']-scores['native'] if resolved else None))
    return dict(complete=complete, arms=arms, paired=paired, decision=decision,
                scope='descriptive feasibility; three independent author draws per arm, not 3000 independent authors; no superiority/noninferiority claim')


def execute_trials(constructor,out,config,collector,stop,snapshot,evaluator):
    trials = [dict(draw=d,arm=a,status='not_attempted') for d,a in ORDER]
    threads = set(); state = dict(status='running',trials=trials)
    guide = Path(config['guide']).read_text(encoding='utf-8')
    native = Path(config['reference_prompt']).read_text(encoding='utf-8')
    snapshot(state)
    for trial in trials:
        if stop():
            state['status']='paused'; break
        policy = constructor()
        folder = out/f"draw_{trial['draw']}_{trial['arm']}"
        text = Path(config['prompts'][trial['arm']]).read_text(encoding='utf-8')
        if text != prompt(native,guide,trial['arm']):
            raise ValueError('frozen prompt differs from task/guide')
        trial['status']='collecting'; snapshot(state)
        try:
            answer,audit=collector(folder,text,stop)
            trial['collection']=audit
            if audit['thread_id'] in threads:
                raise ValueError('fresh thread required')
            threads.add(audit['thread_id'])
        except Exception as exc:
            trial.update(status='collection_failed',error=f'{type(exc).__name__}: {exc}')
            trial['artifact_files']=index_artifacts(folder)
            state['status']='incomplete'; snapshot(state); break
        started=time.perf_counter(); trial['status']='constructing'; snapshot(state)
        try:
            artifact,edit_plan,receipt=policy.construct(trial['arm'],answer)
        except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,SyntaxError,RecursionError) as exc:
            trial.update(status='invalid_proposal',error=f'{type(exc).__name__}: {exc}',
                         construction_seconds=time.perf_counter()-started)
            atomic_json(folder/'invalid.json',dict(error=trial['error'],answer_sha256=file_hash(folder/'answer.md')))
            trial['artifact_files']=index_artifacts(folder)
            snapshot(state); continue
        trial.update(construction_seconds=time.perf_counter()-started,candidate_id=artifact.sha256,
                     numeric_profile=receipt['numeric_profile'],nonzero=receipt['nonzero'],resources=receipt['resources'],
                     source_bytes=len(edit_plan.to_dict()['source'].encode('utf-8')))
        # Native normalized plans are post-hoc archives; never supplied to either writer.
        artifact.save(folder/'candidate.json'); edit_plan.save(folder/'edit.plan.json')
        (folder/'source.wspl').write_text(edit_plan.to_dict()['source'],encoding='utf-8',newline='\n')
        atomic_json(folder/'construction.json',receipt)
        atomic_json(folder/'stored_equations.json',stored_equations(artifact))
        np.savez(folder/'policy.npz',**policy.weights)
        loaded=BreakoutPolicy(folder/'policy.npz',file_hash(folder/'policy.npz'))
        if loaded.tensor_identity()!=policy.tensor_identity() or TensorArtifact.load(folder/'candidate.json').sha256!=artifact.sha256:
            raise ValueError('deployed artifact roundtrip differs')
        atomic_json(folder/'deployment_check.json',frozen_checks(loaded,artifact))
        trial['status']='evaluating'; snapshot(state); started=time.perf_counter()
        records=[]
        for index in range(config['episodes']):
            if stop():
                state['status']='paused'; break
            record=evaluator(loaded,trial,index,folder,stop)
            if record['status']!='complete':
                trial['discarded_partial_episode']=record; state['status']='paused'; break
            record.update(draw=trial['draw'],arm=trial['arm'],index=index)
            with (folder/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            records.append(record); trial['completed_episodes']=len(records)
            if index%25==0:
                snapshot(state)
        if loaded.tensor_identity()!=receipt['tensor_sha256']:
            raise ValueError('deployed weights changed during evaluation')
        trial.update(status='complete' if len(records)==config['episodes'] else 'evaluation_incomplete',
                     summary=summarize(records) if records else None,evaluation_seconds=time.perf_counter()-started)
        # Hash all evidence present after collection and evaluation, including failures elsewhere.
        trial['artifact_files']=index_artifacts(folder)
        snapshot(state)
        if state['status']=='paused':
            break
    if state['status']=='running':
        state['status']='complete'
    state.update(comparisons=comparisons(trials),planned_invocations=6,
                 attempted_invocations=sum(t['status']!='not_attempted' for t in trials),
                 training_updates=0,local_gpu_used=False,formal_status='UNSUPPORTED_TANH')
    snapshot(state); return state


def run(constructor,output_dir,config):
    out=Path(output_dir).resolve()
    receipt=json.loads((out/'source_receipt.json').read_text(encoding='utf-8'))
    if receipt['status']!='verified' or file_hash(out/'approved_manifest.json')!=receipt['manifest_sha256']:
        raise ValueError('approved source receipt required')
    if (config['episodes'],config['max_steps'],config['call_guard_seconds'],config['wall_guard_seconds'])!=(1000,10000,1200,14400):
        raise ValueError('frozen study budgets required')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':
            raise ValueError(key+' must be 1 at startup')
    def verify_inputs():
        for path,expected in config['bound_files'].items():
            if file_hash(path)!=expected:
                raise ValueError('bound file changed: '+path)
        if dependency_identity()['sha256']!=config['dependencies_sha256']:
            raise ValueError('dependency identity differs')
    verify_inputs()
    cli=json.loads(Path(config['local_paths']).read_text(encoding='utf-8'))['codex_cli']
    if file_hash(cli)!=config['cli_sha256']:
        raise ValueError('CLI identity differs')
    atomic_json(out/'dependencies.json',dependency_identity())
    started=time.monotonic(); last={}
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def snapshot(state):
        state.update(active_seconds=time.monotonic()-started,manifest_sha256=receipt['manifest_sha256'])
        last.update(state); atomic_json(out/'status.json',state)
    def collector(folder,text,should_stop):
        return collect(cli,config['catalog'],config['instructions'],folder,text,config['call_guard_seconds'],should_stop)
    def evaluator(policy,trial,index,folder,should_stop):
        from minatar import Environment
        env=Environment('breakout',sticky_action_prob=.1,difficulty_ramping=True)
        if env.env.channels!={'paddle':0,'ball':1,'trail':2,'brick':3}:
            raise ValueError('channel schema differs')
        if index<3:
            env=TraceEnvironment(env)
        result=episode(policy,env,env_seed=1110000+index,action_seed=1210000+index,
                       max_steps=10000,should_stop=should_stop)
        if index<3:
            env.save(folder/f'trajectory_{index}.npz')
        return result
    try:
        final=execute_trials(constructor,out,config,collector,stop,snapshot,evaluator)
        verify_inputs(); atomic_json(out/'results.json',final)
    except BaseException as exc:
        last.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        snapshot(last); atomic_json(out/'results.json',last); raise


if __name__=='__main__':
    raise SystemExit('Use source_consistency.py with an exact user-approved source manifest.')
