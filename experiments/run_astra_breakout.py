"""Source-gated fresh native authoring; never invoke directly."""
import json
import os
from pathlib import Path
import time

from skillforge.codex_one_shot import collect
from skillforge.csail_breakout import dependency_identity, episode, file_hash, summarize
from run_csail_breakout import atomic_json


def run(constructor, output_dir, config):
    output=Path(output_dir).resolve()
    receipt=json.loads((output/'source_receipt.json').read_text())
    if receipt['status']!='verified' or file_hash(output/'approved_manifest.json')!=receipt['manifest_sha256']:
        raise RuntimeError('approved-source receipt required')
    def verify_inputs():
        for path,expected in config['bound_files'].items():
            if file_hash(path)!=expected:raise ValueError('bound file changed: '+path)
        if dependency_identity()['sha256']!=config['dependencies_sha256']:
            raise ValueError('scientific dependency identity differs')
    verify_inputs()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError(key+' must be 1 at process startup')
    if (config['draws'],config['episodes'],config['max_steps'])!=(3,1000,10000):
        raise ValueError('frozen three-draw native screen required')
    import numpy as np
    from minatar import Environment
    started=time.monotonic();trials=[dict(draw=i,status='not_attempted',summary=None) for i in range(3)]
    threads=set();state=dict(status='running',trials=trials,source_manifest_sha256=receipt['manifest_sha256'])
    def snapshot():
        state['active_seconds']=time.monotonic()-started
        atomic_json(output/'status.json',state)
    def stop():
        control=output/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (control.exists() and json.loads(control.read_text()).get('stop'))
    prompt=Path(config['prompt']).read_text(encoding='utf-8')
    cli=json.loads(Path(config['local_paths']).read_text(encoding='utf-8'))['codex_cli']
    if file_hash(cli)!=config['cli_sha256']:raise ValueError('Codex executable identity differs')
    try:
        for trial in trials:
            if stop():state['status']='paused';break
            policy=constructor()  # Also verify before every independent collection.
            i=trial['draw'];trial['status']='collecting';snapshot()
            try:
                answer,audit=collect(cli,config['catalog'],config['instructions'],
                    output/f'draw_{i}',prompt,config['call_guard_seconds'],stop)
                if audit['thread_id'] in threads:raise ValueError('thread ID reused')
                threads.add(audit['thread_id']);trial['collection']=audit
            except Exception as exc:
                trial.update(status='collection_failed',error=f'{type(exc).__name__}: {exc}')
                state['status']='incomplete';break  # No retries/fallback; preserve planned denominator.
            try:parsed=policy.apply_answer(answer)
            except (ValueError,TypeError) as exc:
                trial.update(status='invalid_proposal',error=str(exc));snapshot();continue
            trial['parse']=parsed
            # Native invalid-statement skipping is preserved, even all-zero answers.
            np.savez(output/f'draw_{i}'/'policy.npz',**policy.weights)
            trial['weights_sha256']=file_hash(output/f'draw_{i}'/'policy.npz')
            trial['status']='evaluating';snapshot();records=[];last=time.monotonic()
            for index in range(config['episodes']):
                if stop():state['status']='paused';break
                env=Environment('breakout',sticky_action_prob=.1,difficulty_ramping=True)
                if env.env.channels!={'paddle':0,'ball':1,'trail':2,'brick':3}:raise ValueError('channel schema differs')
                record=episode(policy,env,env_seed=510000+index,action_seed=610000+index,
                    max_steps=10000,should_stop=stop)
                if record['status']!='complete':
                    trial['discarded_partial_episode']=record;state['status']='paused';break
                record.update(draw=i,index=index)
                with (output/'episodes.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
                records.append(record);trial['completed_episodes']=len(records)
                if time.monotonic()-last>30:snapshot();last=time.monotonic()
            if policy.tensor_identity()!=parsed['tensor_sha256']:raise ValueError('frozen weights changed')
            trial['summary']=summarize(records) if records else None
            trial['status']='complete' if len(records)==1000 else 'evaluation_incomplete'
            if state['status']=='paused':break
        if state['status']=='running':state['status']='complete'
        verify_inputs()
    except BaseException:
        state['status']='failed';raise
    finally:
        state.update(planned_draws=3,attempted_draws=sum(t['status']!='not_attempted' for t in trials),
            completed_evaluations=sum(t['status']=='complete' for t in trials),
            scope='exploratory fresh native tanh authoring, no WSPL superiority or formal proof',
            usage='CLI-reported tokens; ChatGPT-backed usage, no enforceable dollar/output-token cap',
            gpu_used=False,training_updates=0)
        snapshot();atomic_json(output/'results.json',state)


if __name__=='__main__':raise SystemExit('Use source_consistency.py with exact user-approved manifest.')
