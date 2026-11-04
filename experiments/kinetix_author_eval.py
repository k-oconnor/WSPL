"""WSL CPU worker; requires a live approved-source authoring receipt."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from source_consistency import verify
from skillforge.kinetix_replay import make_environment,sample_action,sha,SHAPES
from skillforge.wsp_kinetix import KinetixRepair
from run_kinetix_replay import save


def evaluate(policy,env,initial,params,folder,stop,episodes=1024):
    import jax
    import jax.numpy as jnp
    rows=[]
    with (folder/'episodes.jsonl').open('x',encoding='utf-8') as log:
        for index in range(episodes):
            rng=np.random.default_rng(10100000+index)
            obs,state=env.reset(jax.random.PRNGKey(index),params,override_reset_state=initial)
            total=0.;trace=[];outcome=None
            for step in range(256):
                if stop():
                    save(folder/'evaluation.json',dict(status='incomplete',completed=len(rows),partial_episode=index,partial_steps=step))
                    return
                inputs=np.asarray(obs,dtype=np.float32)
                action=sample_action(policy.logits(inputs),rng)
                previous=float(state.last_distance)
                obs,state,reward,done,info=env.step(jax.random.PRNGKey(step),state,jnp.asarray(action),params)
                reward=float(reward);done=bool(done)
                finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(state))
                if index<3:trace.append(dict(step=step,observation=inputs.tolist(),action=action.tolist(),reward=reward if np.isfinite(reward) else None,done=done))
                if not finite or not np.isfinite(reward):outcome='invalid';break
                total+=reward
                if done:
                    shaping=0. if previous==-1 else .2*(previous-float(info['distance']))
                    outcome='success' if bool(info['GoalR']) else ('red' if reward-shaping<-.5 else 'timeout')
                    break
            row=dict(index=index,seed=10100000+index,steps=step+1,outcome=outcome or 'timeout',return_=total)
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if index<3:save(folder/f'trajectory_{index}.json',trace)
            if index%16==0:save(folder/'evaluation.json',dict(status='running',completed=len(rows)))
    counts={o:sum(r['outcome']==o for r in rows) for o in ('success','red','timeout','invalid')}
    save(folder/'evaluation.json',dict(status='complete',completed=len(rows),outcomes=counts,
        success_rate=counts['success']/len(rows),mean_return=float(np.mean([r['return_'] for r in rows])),mean_length=float(np.mean([r['steps'] for r in rows]))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--folder',required=True);p.add_argument('--digest',required=True);p.add_argument('--weights-sha256',required=True)
    a=p.parse_args();out=Path(a.run).resolve();folder=Path(a.folder).resolve()
    if not out.is_relative_to(ROOT/'results') or not folder.is_relative_to(out):raise ValueError('output containment failed')
    receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or receipt['manifest_sha256']!=a.digest:raise ValueError('live source receipt required')
    manifest=verify(ROOT,out/'approved_manifest.json',a.digest)
    registered=json.loads((ROOT/'experiments/source_pins/approved.json').read_text())['models'].get(manifest['model_id'])
    if not registered or registered['sha256']!=a.digest:raise ValueError('independent registry approval required')
    for path,digest in manifest['runner_config']['linux_resources'].items():
        if sha(path)!=digest:raise ValueError('Linux runtime changed: '+path)
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('CPU required')
    if sha(folder/'policy.npz')!=a.weights_sha256:raise ValueError('weights identity differs')
    policy=object.__new__(KinetixRepair)
    with np.load(folder/'policy.npz',allow_pickle=False) as z:policy.weights={k:z[k] for k in z.files}
    if set(policy.weights)!=set(SHAPES):raise ValueError('tensor names differ')
    for k,w in policy.weights.items():
        if w.shape!=SHAPES[k] or w.dtype!=np.float32 or not np.isfinite(w).all() or (np.abs(w)>1e6).any():raise ValueError('FP32 actor required')
        w.flags.writeable=False
    env,initial,params=make_environment(manifest['runner_config']['level'])
    started=time.monotonic()
    def stop():
        control=out/'control.json'
        return time.monotonic()-started>7200 or (control.exists() and json.loads(control.read_text()).get('stop'))
    evaluate(policy,env,initial,params,folder,stop)
    if sha(folder/'policy.npz')!=a.weights_sha256:raise ValueError('weights changed')
    verify(ROOT,out/'approved_manifest.json',a.digest)
    for path,digest in manifest['runner_config']['linux_resources'].items():
        if sha(path)!=digest:raise ValueError('Linux runtime changed during evaluation: '+path)

if __name__=='__main__':main()
