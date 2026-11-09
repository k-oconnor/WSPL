"""Development-reward recipe selection, followed by frozen learning curves."""
import json
from pathlib import Path
import time
import numpy as np
from skillforge.arc_eval import model_hash
from skillforge.selective_key import features,evaluate,validate_key
from skillforge.key_learning import DEFAULT,fit_key,nested_indices,parse_recipe,reward,best_trial
from skillforge.codex_one_shot import collect
from skillforge.reliable_json import save,status as save_status
from skillforge.go_replay import sha

SIZES=(16,64,256,1024)
SEEDS=(17,29,43)
GRID=[dict(DEFAULT),dict(DEFAULT,negative_weight=10.,retention_weight=10.,worst_negative=3.),
      dict(DEFAULT,ridge=.01,head_variance=.1),dict(DEFAULT,negative_weight=10.,retention_weight=10.,ridge=.01,worst_negative=3.,head_variance=.1)]


def make_prompt(guide,packet,history,step):
    text=guide+'\nACQUISITION PACKET:\n'+json.dumps(packet)+'\nVISIBLE TRANSCRIPT:\n'+json.dumps(history)
    text+=f'\nProposal {step+1}/4. Return exactly one recipe JSON object.'
    if len(text.encode())>800000:raise ValueError('prompt byte budget')
    return text


def run(constructor,output_dir,config):
    import torch
    out=Path(output_dir).resolve();receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('source receipt')
    def verify_resources():
        for path,h in config['resources'].items():
            if sha(path)!=h:raise ValueError('resource changed: '+path)
    verify_resources();started=time.monotonic();state=dict(status='loading',manifest_sha256=receipt['manifest_sha256'])
    def snapshot():state['active_seconds']=time.monotonic()-started;save_status(out/'status.json',state)
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def check():
        if stop():raise RuntimeError('stopped/guard')
    try:
        study=constructor();before=model_hash(study.model)
        if before!=config['model_sha256']:raise ValueError('base model mismatch')
        with np.load(config['parent_bank'],allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
        if np.argwhere(parent['alpha']>0).tolist()!=[[2,0]] or parent['alpha'][2,0]!=.75:raise ValueError('parent slot')
        initial=parent['K'][2,0].copy();bound=10*float(parent['K_scale'][2,0])
        rows={split:json.loads(Path(config[split]).read_text()) for split in ('train','development')}
        caches={};base_reports={}
        for split in ('train','development'):
            q=[];z=[];base=[];state.update(status='caching',split=split);snapshot()
            for i,row in enumerate(rows[split]):
                check();qq,zz,bb=features(study,[row]);q.append(qq[0]);z.append(zz[0]);base.append(bb[0])
                if i%32==0:state.update(examples_completed=i+1,examples_total=len(rows[split]));snapshot()
            q=np.asarray(q);z=np.asarray(z);y=np.asarray([r['category']=='conduction' for r in rows[split]]);correct=np.asarray([r['correct'] for r in base])
            caches[split]=dict(q=q,z=z,y=y,correct=correct,base=base)
            np.savez_compressed(out/(split+'_cache.npz'),q=q,logz=z,applicable=y,base_correct=correct)
            save(out/(split+'_baseline.json'),base)
        dev=caches['development'];train=caches['train']
        base_reports['development']=evaluate(study,rows['development'],None,'off',dev['base'],check)
        if base_reports['development']['records']!=dev['base']:raise ValueError('baseline parity')
        plans={seed:nested_indices([r['category'] for r in rows['train']],seed,SIZES) for seed in SEEDS}
        save(out/'subset_indices.json',{str(seed):{str(n):idx.tolist() for n,idx in plan.items()} for seed,plan in plans.items()})
        def fit(recipe,indices,folder):
            folder.mkdir();check()
            key,trace=fit_key(train['q'][indices],train['z'][indices],train['y'][indices],train['correct'][indices],initial,bound,recipe,config['optimizer_steps'],check)
            bank={k:v.copy() for k,v in parent.items()};bank['K'][2,0]=key;validate_key(bank,parent)
            # Only the key is new; shared frozen value/base are hash-referenced, not duplicated per fit.
            np.savez_compressed(folder/'key.npz',K=key)
            save(folder/'fit.json',dict(recipe=recipe,steps=config['optimizer_steps'],training_ids=[rows['train'][int(i)]['id'] for i in indices],trace=trace,
                parent_sha256=sha(config['parent_bank']),manifest_sha256=receipt['manifest_sha256']))
            return bank
        def tune(recipe,folder):
            bank=fit(recipe,plans[17][256],folder)
            result=evaluate(study,rows['development'],bank,'retrieval',dev['base'],check)
            save(folder/'development.json',result)
            return result,reward(result,base_reports['development'])
        grid=[]
        for i,recipe in enumerate(GRID):
            state.update(status='fixed_grid',trial=i);snapshot();result,rew=tune(recipe,out/f'grid_{i}')
            grid.append(dict(step=i,status='accepted',recipe=recipe,reward=rew,metrics=result['metrics']))
        grid_selection=best_trial(grid);save(out/'grid_trials.json',grid)
        guide=Path(config['guide']).read_text();cli=json.loads(Path(config['local_paths']).read_text())['codex_cli']
        packet=dict(training_count=len(rows['train']),development_count=len(rows['development']),
            training_sample=[rows['train'][int(i)] for i in plans[17][16]],
            development_baseline=base_reports['development']['metrics'],starting_recipe=DEFAULT,
            scope='Recipe selection only; fixed64parameter key learner. Grid outcomes withheld from author.')
        history=[];trials=[];threads=set()
        for i in range(4):
            state.update(status='authoring_recipe',trial=i);snapshot();folder=out/f'astra_{i}';t=time.monotonic();last=[0.]
            def heartbeat():
                now=time.monotonic()
                if now-last[0]>5:last[0]=now;state.update(call_active_seconds=now-t);snapshot()
                return stop()
            answer,audit=collect(cli,config['catalog'],config['instructions'],folder,make_prompt(guide,packet,history,i),1200,heartbeat)
            if audit['thread_id'] in threads:raise ValueError('reused thread')
            threads.add(audit['thread_id']);entry=dict(step=i,collection=audit)
            try:recipe=parse_recipe(answer)
            except (ValueError,TypeError) as e:entry.update(status='rejected',error=str(e));feedback=dict(error=str(e))
            else:
                result,rew=tune(recipe,folder/'fit');entry.update(status='accepted',recipe=recipe,reward=rew,metrics=result['metrics'])
                feedback=dict(reward=rew,metrics=result['metrics'])
            save(folder/'feedback.json',feedback);trials.append(entry);history.append(dict(assistant=answer,feedback=feedback))
        selected=dict(fixed_grid=grid_selection,astra=best_trial(trials));save(out/'astra_trials.json',trials);save(out/'selected_recipes.json',selected)
        frozen=[]
        for arm,selection in selected.items():
            for n in SIZES:
                for seed in ((17,) if n==1024 else SEEDS):
                    name=f'{arm}_n{n}_subset{seed}';state.update(status='fitting_curve',arm=arm,n=n,subset_seed=seed);snapshot()
                    fit(selection['recipe'],plans[seed][n],out/name)
                    frozen.append(dict(name=name,arm=arm,n=n,subset_seed=seed if n<1024 else None,key_sha256=sha(out/name/'key.npz')))
        save(out/'frozen_curve.json',frozen)
        # Final cases are loaded only after recipes AND all20curve keys freeze.
        final_rows=json.loads(Path(config['final']).read_text());final_base=[];state.update(status='evaluating_final');snapshot()
        for i,row in enumerate(final_rows):
            check();_,_,b=features(study,[row]);final_base.append(b[0])
        save(out/'final_baseline.json',final_base)
        reports={}
        for name,bank,mode in [('untouched',None,'off'),('original_key',parent,'retrieval'),('forced',parent,'forced')]:
            reports[name]=evaluate(study,final_rows,bank,mode,final_base,check);save(out/(name+'_final.json'),reports[name])
        for entry in frozen:
            name=entry['name'];path=out/name/'key.npz';assert sha(path)==entry['key_sha256']
            with np.load(path,allow_pickle=False) as z:key=z['K']
            bank={k:v.copy() for k,v in parent.items()};bank['K'][2,0]=key;validate_key(bank,parent)
            state.update(status='evaluating_final',candidate=name);snapshot()
            reports[name]=evaluate(study,final_rows,bank,'retrieval',final_base,check);save(out/name/'final.json',reports[name])
        for entry in frozen:
            if sha(out/entry['name']/'key.npz')!=entry['key_sha256']:raise ValueError('frozen key changed')
        if model_hash(study.model)!=before:raise ValueError('base weights changed')
        verify_resources();state.update(status='complete',selected_recipes=selected,curve=frozen,final_metrics={k:v['metrics'] for k,v in reports.items()},
            model_sha256=before,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
