"""CPU post-hoc evaluation of all saved edits; no new authoring or fitting."""
import json
import os
from pathlib import Path
import time
import numpy as np
from skillforge.go_replay import sha,make_batch_runner
from skillforge.go_edit_sequence import validate_batch,paired_difference
from run_diffusion_replay import save
from run_go_replay import interval


def run(constructor,output_dir,config):
    import jax
    import jax.numpy as jnp
    out=Path(output_dir);receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('approved receipt required')
    if any(d.platform!='cpu' for d in jax.devices()) or jax.config.jax_enable_x64:raise ValueError('CPU FP32 required')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('thread configuration differs')
    def verify_resources():
        for path,h in config['resources'].items():
            if sha(path)!=h:raise ValueError('resource changed: '+path)
    verify_resources();study=constructor();play=make_batch_runner();started=time.monotonic()
    state=dict(status='running',completed_games=0,manifest_sha256=receipt['manifest_sha256'])
    outcomes={};evaluations={};historical_batches=0
    def snapshot():
        state['active_seconds']=time.monotonic()-started;save(out/'status.json',state)
    snapshot()
    try:
        opponents=[jax.tree.map(jnp.asarray,w) for w in study.opponents]
        for name,weights in study.actors.items():
            actor=jax.tree.map(jnp.asarray,weights);folder=out/name;folder.mkdir();summaries={};wins_by_opponent=[]
            for opponent,w in enumerate(opponents):
                win_parts=[];lengths=[];color_wins=[0,0]
                for start in range(0,2000,32):
                    control=out/'control.json'
                    if time.monotonic()-started>config['wall_guard_seconds'] or (control.exists() and json.loads(control.read_text()).get('stop')):
                        raise RuntimeError('user stop or wall guard; incomplete')
                    ids=np.arange(start,min(start+32,2000),dtype=np.int32);colors=ids%2;seeds=11300000+opponent*100000+ids
                    keys=jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds,dtype=jnp.uint32))
                    result,trace=play(actor,w,keys,jnp.asarray(colors));result=jax.tree.map(np.asarray,result);trace=jax.tree.map(np.asarray,trace)
                    validate_batch(result,ids)
                    filename=f's{opponent}_batch_{start:04d}.npz'
                    payload=dict(episode_ids=ids,seeds=seeds,actor_colors=colors,**result,**trace)
                    np.savez_compressed(folder/filename,**payload)
                    if name=='edit_6':
                        with np.load(Path(config['historical_final'])/filename,allow_pickle=False) as old:
                            if set(old.files)!=set(payload) or any(not np.array_equal(old[k],v) for k,v in payload.items()):
                                raise ValueError('final checkpoint replay differs from archived evaluation')
                        historical_batches+=1
                    won=result['reward']>0;win_parts.append(won);lengths.extend(result['steps'].tolist())
                    for c in (0,1):color_wins[c]+=int(won[colors==c].sum())
                    state.update(candidate=name,opponent=opponent,completed_games=state['completed_games']+len(ids));snapshot()
                win=np.concatenate(win_parts);wins_by_opponent.append(win)
                summaries[str(opponent)]=dict(games=2000,wins=int(win.sum()),win_rate=float(win.mean()),
                    ci95_wilson=interval(int(win.sum()),2000),color_wins=color_wins,mean_length=float(np.mean(lengths)))
            outcomes[name]=np.stack(wins_by_opponent)
            evaluations[name]=dict(summaries=summaries,mean_win_rate=float(outcomes[name].mean()))
            state['evaluations']=evaluations;snapshot()
        contrasts={f'edit_{i}_minus_edit_{i-1}':paired_difference(outcomes[f'edit_{i-1}'],outcomes[f'edit_{i}']) for i in range(1,7)}
        verify_resources()
        state.update(status='complete',contrasts=contrasts,historical_final_batches_matched=historical_batches,
            primary_descriptive_contrasts=['edit_6_minus_edit_5','edit_5_minus_edit_4'])
    except BaseException as e:
        state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:
        snapshot();save(out/'results.json',state)
