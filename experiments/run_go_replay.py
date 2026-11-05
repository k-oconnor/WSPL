"""Source-gated released-policy Go replay, CPU only, no authoring or training."""
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from skillforge.go_replay import sha,make_batch_runner
from run_diffusion_replay import save


def interval(wins,n):
    if not n:return None
    p=wins/n;z=1.959963984540054;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [mid-half,mid+half]


def run(constructor,output_dir,config):
    import jax
    import jax.numpy as jnp
    import pgx
    out=Path(output_dir);receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('approved receipt required')
    if (jax.__version__,np.__version__,pgx.__version__)!=('0.4.38','1.26.4','2.6.0'):raise ValueError('runtime differs')
    if any(d.platform!='cpu' for d in jax.devices()) or jax.config.jax_enable_x64:raise ValueError('CPU FP32 execution required')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('single-thread environment required')
    if config['games_per_match']!=2000 or config['batch_size']!=32:raise ValueError('budget differs')
    def verify():
        for p,h in config['resources'].items():
            if sha(p)!=h:raise ValueError('resource differs: '+p)
    verify();study=constructor();play=make_batch_runner()
    actors={name:jax.tree.map(jnp.asarray,w) for name,w in study.actors.items()}
    for name,w in study.actors.items():
        arrays={('L'+str(i) if i<7 else 'out')+'.'+suffix:a for i,pair in enumerate(w,1) for suffix,a in zip(('W','b'),pair)}
        np.savez_compressed(out/(name+'_actor.npz'),**arrays)
    started=time.monotonic();records=[];status='running'
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    save(out/'status.json',dict(status=status,completed_games=0))
    with (out/'episodes.jsonl').open('w') as log:
        for opponent in range(3):
            for name in ('budget_0','budget_50'):
                pair=f'{name}_vs_s{opponent}';folder=out/pair;folder.mkdir()
                for start in range(0,2000,32):
                    if stop():status='paused';break
                    ids=np.arange(start,min(start+32,2000),dtype=np.int32);colors=ids%2
                    # Independent episode keys; same keys/seats for both authored actors.
                    seeds=config['seed_base']+opponent*100000+ids
                    keys=jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds,dtype=jnp.uint32))
                    result,trace=play(actors[name],actors[f'ppo_same_size_lr3e-5_s{opponent}'],keys,jnp.asarray(colors))
                    result=jax.tree.map(np.asarray,result);trace=jax.tree.map(np.asarray,trace)
                    if np.any(result['invalid']) or not np.all(result['terminal']) or not np.isin(result['reward'],[-1,1]).all():
                        np.savez_compressed(folder/f'failed_batch_{start}.npz',**result,**trace)
                        raise ValueError('invalid game execution; stopped without scoring this batch')
                    archive=folder/f'batch_{start:04d}.npz'
                    np.savez_compressed(archive,episode_ids=ids,seeds=seeds,actor_colors=colors,**result,**trace)
                    for i,episode in enumerate(ids):
                        steps=int(result['steps'][i]);actions=trace['actions'][:,i]
                        if not 2<=steps<=162 or np.count_nonzero(actions>=0)!=steps:raise ValueError('move accounting differs')
                        row=dict(actor=name,opponent=opponent,episode=int(episode),seed=int(seeds[i]),actor_color=int(colors[i]),
                            win=bool(result['reward'][i]>0),steps=steps,
                            reason='repetition' if bool(result['psk'][i]) else ('passes' if result['passes'][i]>=2 else 'horizon'))
                        records.append(row);log.write(json.dumps(row)+'\n')
                    log.flush()
                    save(out/'status.json',dict(status='running',match=pair,completed_in_match=int(ids[-1])+1,completed_games=len(records),
                        active_seconds=time.monotonic()-started,last_batch_sha256=sha(archive)))
                if status=='paused':break
            if status=='paused':break
    summaries={}
    for opponent in range(3):
        for name in ('budget_0','budget_50'):
            rows=[r for r in records if r['actor']==name and r['opponent']==opponent];n=len(rows);wins=sum(r['win'] for r in rows)
            summaries[f'{name}_vs_s{opponent}']=dict(games=n,wins=wins,win_rate=wins/n if n else None,ci95_wilson=interval(wins,n),
                colors={str(c):dict(games=sum(r['actor_color']==c for r in rows),wins=sum(r['win'] for r in rows if r['actor_color']==c)) for c in (0,1)},
                reasons={k:sum(r['reason']==k for r in rows) for k in ('repetition','passes','horizon')})
    verify()
    final=dict(status='complete' if status=='running' else status,completed_games=len(records),summaries=summaries,
        active_seconds=time.monotonic()-started,manifest_sha256=receipt['manifest_sha256'],scope='Statistical artifact replay with pinned reconstructed evaluator; no fresh authoring.')
    save(out/'results.json',final);save(out/'status.json',final)
