"""CPU evaluation worker, requiring the approved live parent study receipt."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify
from skillforge.go_authoring import read_arrays
from skillforge.go_replay import sha,load_actor,make_batch_runner
from run_diffusion_replay import save
from run_go_replay import interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--study-output',required=True);p.add_argument('--candidate',required=True);a=p.parse_args()
    out=Path(a.study_output);receipt=json.loads((out/'source_receipt.json').read_text())
    model_id=json.loads((out/'approved_manifest.json').read_text())['model_id']
    if model_id not in ('go-authoring-v1','go-authoring-resume-v1'):raise ValueError('unexpected study')
    registry=json.loads((ROOT/'experiments/source_pins/approved.json').read_text())['models'][model_id]
    if receipt['status']!='verified' or receipt['manifest_sha256']!=registry['sha256']:raise ValueError('live independently registered study required')
    m=verify(ROOT,out/'approved_manifest.json',registry['sha256']);config=m['runner_config']
    for path,h in config['linux_resources'].items():
        if sha(path)!=h:raise ValueError('Linux resource differs: '+path)
    frozen=json.loads((out/'frozen_candidates.json').read_text())
    if len(frozen)!=7 or a.candidate not in frozen:raise ValueError('all six candidates and zero control must be frozen before any games')
    for name,row in frozen.items():
        if sha(out/row['path'])!=row['sha256']:raise ValueError('frozen candidate changed')
    import jax
    import jax.numpy as jnp
    if any(d.platform!='cpu' for d in jax.devices()) or jax.config.jax_enable_x64:raise ValueError('CPU FP32 required')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('thread configuration differs')
    candidate=read_arrays(out/frozen[a.candidate]['path']);actor=tuple((jnp.asarray(candidate['W'+str(i)]),jnp.asarray(candidate['b'+str(i)])) for i in range(1,8))
    dest=out/a.candidate/'evaluation';dest.mkdir(exist_ok=False);play=make_batch_runner();started=time.monotonic();summaries={}
    for opponent in range(3):
        w=load_actor(Path(config['opponent_root'])/f'ppo_same_size_lr3e-5_s{opponent}');w=jax.tree.map(jnp.asarray,w);wins=0;color_wins=[0,0];lengths=[]
        for start in range(0,2000,32):
            control=out/'control.json'
            if control.exists() and json.loads(control.read_text()).get('stop'):raise RuntimeError('user stop during evaluation; no auto-resume')
            ids=np.arange(start,min(start+32,2000),dtype=np.int32);colors=ids%2;seeds=11300000+opponent*100000+ids
            keys=jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds,dtype=jnp.uint32))
            result,trace=play(actor,w,keys,jnp.asarray(colors));result=jax.tree.map(np.asarray,result);trace=jax.tree.map(np.asarray,trace)
            if np.any(result['invalid']) or not np.all(result['terminal']) or not np.isin(result['reward'],[-1,1]).all():raise ValueError('invalid policy execution')
            np.savez_compressed(dest/f's{opponent}_batch_{start:04d}.npz',episode_ids=ids,seeds=seeds,actor_colors=colors,**result,**trace)
            won=result['reward']>0;wins+=int(won.sum());lengths.extend(result['steps'].tolist())
            for c in (0,1):color_wins[c]+=int(won[colors==c].sum())
            save(dest/'status.json',dict(status='running',opponent=opponent,games_completed=opponent*2000+int(ids[-1])+1,active_seconds=time.monotonic()-started))
        summaries[str(opponent)]=dict(games=2000,wins=wins,win_rate=wins/2000,ci95_wilson=interval(wins,2000),color_wins=color_wins,mean_length=float(np.mean(lengths)))
    result=dict(status='complete',summaries=summaries,mean_win_rate=float(np.mean([r['win_rate'] for r in summaries.values()])),active_seconds=time.monotonic()-started)
    save(dest/'results.json',result);save(dest/'status.json',result)


if __name__=='__main__':main()
