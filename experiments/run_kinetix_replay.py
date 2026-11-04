"""Source-gated fixed-artifact Kinetix replay; no training or authoring."""
import json
import os
from pathlib import Path
import time
import numpy as np
from skillforge.kinetix_replay import sha,make_environment,sample_action


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


def verify_resources(config):
    for path,digest in config['resources'].items():
        if sha(path)!=digest:raise ValueError('resource identity differs: '+path)


def run(constructor,output_dir,config):
    out=Path(output_dir)
    receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:
        raise RuntimeError('approved source receipt required')
    verify_resources(config)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError(name+' must be 1')
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('CPU only')
    import jax
    import jax.numpy as jnp
    if jax.__version__!='0.4.38' or any(d.platform!='cpu' for d in jax.devices()):raise ValueError('runtime differs')
    if config['episodes_per_arm']!=1024:raise ValueError('fixed 1024-episode budget required')
    study=constructor();save(out/'reconstruction.json',study.reconstruction)
    for mode,policy in study.policies.items():np.savez_compressed(out/(mode+'_actor.npz'),**policy.weights)
    env,initial,params=make_environment(config['level'])
    started=time.monotonic();last=started;records=[];stopped=False
    with (out/'episodes.jsonl').open('w',encoding='utf-8') as log:
        for episode in range(1024):
            for mode,policy in study.policies.items():
                seed=config['action_seed_base']+episode;rng=np.random.default_rng(seed)
                obs,state=env.reset(jax.random.PRNGKey(episode),params,override_reset_state=initial)
                total=0.;trace=[];outcome=None;partial=False
                for step in range(256):
                    control=out/'control.json'
                    if time.monotonic()-started>config['wall_guard_seconds'] or (control.exists() and json.loads(control.read_text()).get('stop')):
                        partial=True;stopped=True;break
                    observed=np.asarray(obs,dtype=np.float32)
                    if observed.shape!=(679,) or not np.isfinite(observed).all():raise ValueError('bad observation')
                    action=sample_action(policy.logits(observed),rng)
                    previous_distance=float(state.last_distance)
                    obs,state,reward,done,info=env.step(jax.random.PRNGKey(step),state,jnp.asarray(action),params)
                    reward=float(reward);done=bool(done)
                    finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(state))
                    if episode<3:trace.append(dict(step=step,observation=observed.tolist(),action=action.tolist(),reward=reward if np.isfinite(reward) else None,done=done))
                    if not finite or not np.isfinite(reward):outcome='invalid';break
                    total+=reward
                    if done:
                        shaping=0. if previous_distance==-1 else .2*(previous_distance-float(info['distance']))
                        sparse=reward-shaping
                        outcome='success' if bool(info['GoalR']) else ('red' if sparse<-.5 else 'timeout')
                        break
                row=dict(mode=mode,episode=episode,seed=seed,steps=len(trace) if partial and episode<3 else (step if partial else step+1),
                         outcome='incomplete' if partial else (outcome or 'timeout'),return_=total,policy_sha256=policy.tensor_sha256)
                records.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if episode<3:save(out/f'{mode}_{episode}_trajectory.json',trace)
                if time.monotonic()-last>30 or stopped:
                    save(out/'status.json',dict(status='paused' if stopped else 'running',completed=sum(r['outcome']!='incomplete' for r in records),active_seconds=time.monotonic()-started));last=time.monotonic()
                if stopped:break
            if stopped:break
    summaries={}
    for mode in study.policies:
        rows=[r for r in records if r['mode']==mode and r['outcome']!='incomplete']
        summaries[mode]=dict(episodes=len(rows),outcomes={o:sum(r['outcome']==o for r in rows) for o in ('success','red','timeout','invalid')},
            mean_return=float(np.mean([r['return_'] for r in rows])) if rows else None,
            mean_length=float(np.mean([r['steps'] for r in rows])) if rows else None)
    verify_resources(config)
    result=dict(status='paused' if stopped else 'complete',summaries=summaries,active_seconds=time.monotonic()-started,
        manifest_sha256=receipt['manifest_sha256'],inference='NumPy float64 actor; FP32 simulator observations/physics; PCG64 categorical')
    save(out/'results.json',result);save(out/'status.json',result)
