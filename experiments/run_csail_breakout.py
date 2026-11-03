"""Approved-source native tanh artifact replay; no model calls or training."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

from skillforge.csail_breakout import dependency_identity, episode, file_hash, summarize


def atomic_json(path,value):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    os.replace(tmp,path)


def run(constructor, output_dir, config):
    output = Path(output_dir)
    receipt = json.loads((output/'source_receipt.json').read_text())
    if receipt['status']!='verified' or file_hash(output/'approved_manifest.json')!=receipt['manifest_sha256']:
        raise RuntimeError('approved-source receipt required')
    for path,expected in config['auxiliary_files'].items():
        if file_hash(path)!=expected:raise ValueError('auxiliary identity differs: '+path)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(key)!='1': raise ValueError(key+' must be 1 before process start')
    import numpy as np
    from minatar import Environment
    identity = dependency_identity()
    if identity['sha256'] != config['dependencies_sha256']: raise ValueError('scientific dependency identity differs')
    atomic_json(output/'dependencies.json',identity)
    bundle = Path(config['bundle'])
    if file_hash(bundle/'bundle.json')!=config['bundle_sha256']: raise ValueError('reference bundle differs')
    resource = json.loads((bundle/'bundle.json').read_text())
    for f in resource['files']:
        relative = Path(f['path'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('invalid reference path')
        if file_hash(bundle/relative)!=f['sha256']: raise ValueError('reference file changed: '+f['path'])
    atomic_json(output/'reference_bundle.json',resource)
    settings = {k:config[k] for k in ('episodes_per_arm','max_steps','env_seed_base','action_seed_base','wall_guard_seconds')}
    if any(type(x) is not int or x<1 for x in settings.values()): raise ValueError('positive integer settings required')
    if settings['episodes_per_arm']!=1000 or settings['max_steps']!=10000:
        raise ValueError('native replay requires 1000 episodes/10000-step horizon')
    if settings['env_seed_base']==settings['action_seed_base']: raise ValueError('distinct RNG seed streams required')
    started = time.monotonic()
    summaries = {}
    status = 'complete'
    reason = None
    last_progress = started
    snapshot_at = started
    state = dict(status='running',completed={})
    def should_stop():
        nonlocal reason
        if time.monotonic()-started > settings['wall_guard_seconds']:
            reason='wall_guard';return True
        control=output/'control.json'
        if control.exists() and json.loads(control.read_text()).get('stop'):
            reason='user_stop';return True
        return False
    def snapshot():
        atomic_json(output/'status.json',{**state,'active_seconds':time.monotonic()-started})
    try:
        for arm in ('released','uniform'):
            policy = constructor()
            if arm=='uniform':policy.uniform_control()
            tensor_hash=policy.tensor_identity()
            records=[]
            state.update(arm=arm,next_episode=0)
            snapshot()
            for index in range(settings['episodes_per_arm']):
                if should_stop():status='paused';break
                env=Environment('breakout',sticky_action_prob=0.1,difficulty_ramping=True)
                if env.env.channels!={'paddle':0,'ball':1,'trail':2,'brick':3}:
                    raise ValueError('environment channel schema differs')
                record=episode(policy,env,env_seed=settings['env_seed_base']+index,
                    action_seed=settings['action_seed_base']+index,max_steps=settings['max_steps'],should_stop=should_stop)
                if record['status']!='complete':
                    state['discarded_partial_episode']=record;status='paused';break
                record.update(arm=arm,index=index)
                with (output/'episodes.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
                    stream.write(json.dumps(record,allow_nan=False)+'\n')
                records.append(record)
                state['completed'][arm]=len(records)
                state['next_episode']=index+1
                now=time.monotonic()
                if now-last_progress>30:
                    snapshot();print(json.dumps(dict(event='progress',arm=arm,episodes=len(records),mean_so_far=float(np.mean([r['reward'] for r in records])),active_seconds=now-started)),flush=True)
                    last_progress=now
                if now-snapshot_at>1800:snapshot();snapshot_at=now
            if policy.tensor_identity()!=tensor_hash:raise ValueError('frozen policy tensors changed')
            summaries[arm]=dict(summary=summarize(records) if records else None,
                                tensor_sha256=tensor_hash,policy_mode=policy.mode,
                                complete=len(records)==settings['episodes_per_arm'])
            if status!='complete':break
        if dependency_identity()['sha256']!=identity['sha256']:raise ValueError('dependencies changed during evaluation')
        for f in resource['files']:
            if file_hash(bundle/f['path'])!=f['sha256']:raise ValueError('reference changed during evaluation')
        for path,expected in config['auxiliary_files'].items():
            if file_hash(path)!=expected:raise ValueError('auxiliary changed during evaluation: '+path)
        published=json.loads((bundle/'exp5_write_from_scratch/eval/minatar_eval.json').read_text())['breakout']['hand']
        comparison=None
        measured=summaries.get('released',{}).get('summary')
        if status=='complete' and measured:
            combined=math.sqrt(measured['se']**2+published['se']**2)
            comparison=dict(published=published,mean_difference=measured['mean']-published['mean'],
                            difference_over_combined_se=(measured['mean']-published['mean'])/combined if combined else None,
                            inference='descriptive agreement check; original seeds/evaluator not released, not bitwise replication')
        result=dict(status=status,stop_reason=reason,settings=settings,arms=summaries,
                    comparison=comparison,active_seconds=time.monotonic()-started,
                    source_manifest_sha256=receipt['manifest_sha256'],dependencies_sha256=identity['sha256'],
                    scope='empirical native tanh artifact replay, not new Astra authoring, WSPL advantage or formal proof',
                    numerical_profile='NumPy float32 tanh/matmul, float64 softmax, PCG64 action RNG; fresh seeded environment per episode',
                    api_calls=0,training_updates=0,gpu_used=False)
        atomic_json(output/'results.json',result)
        state.update(status=status,stop_reason=reason)
    except BaseException:
        state['status']='failed';raise
    finally:snapshot()


if __name__=='__main__':
    raise SystemExit('Use experiments/source_consistency.py with an exact approved source manifest.')
