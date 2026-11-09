"""Frozen ARC validation transfer; no authoring, fitting, or test-split access."""
import json,time,re
from pathlib import Path
import numpy as np
from skillforge.arc_eval import model_hash
from skillforge.arc_attention import score_boundary
from skillforge.go_replay import sha
from skillforge.reliable_json import save,status


def summarize(records,base):
    old={r['id']:r for r in base}
    return dict(n=len(records),correct=sum(r['correct'] for r in records),
                raw_correct=sum(r['raw_correct'] for r in records),
                corrected=[r['id'] for r in records if r['correct'] and not old[r['id']]['correct']],
                broken=[r['id'] for r in records if not r['correct'] and old[r['id']]['correct']])


def run(constructor,output_dir,config):
    out=Path(output_dir);started=time.monotonic()
    receipt=json.loads((out/'source_receipt.json').read_text());h=receipt['manifest_sha256']
    if sha(out/'approved_manifest.json')!=h:raise ValueError('receipt')
    def resources():
        for p,d in config['resources'].items():
            if sha(p)!=d:raise ValueError('resource changed: '+p)
    resources();state=dict(status='loading',manifest_sha256=h)
    def check():
        state['active_seconds']=time.monotonic()-started
        status(out/'status.json',state)
        control=out/'control.json'
        if state['active_seconds']>config['wall_guard_seconds'] or (control.exists() and json.loads(control.read_text()).get('stop')):
            raise RuntimeError('stop/guard')
    try:
        study=constructor();before=model_hash(study.model)
        if before!=config['model_sha256']:raise ValueError('model identity')
        def load(p):
            with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
        parent=load(config['parent_bank']);ridge_archive=load(config['ridge_bank'])
        # The ridge archive also contains disclosed training diagnostics; only
        # the original bank fields participate in inference.
        ridge={k:ridge_archive[k] for k in parent}
        banks={'untouched':None,'forced':parent,'original_key':parent,'ridge_key':ridge}
        for name,path in config['curve_keys'].items():
            b={k:v.copy() for k,v in parent.items()};b['K'][2,0]=load(path)['K'];banks[name]=b
        for name,b in banks.items():
            if b is None:continue
            if set(b)!=set(parent):raise ValueError('bank fields differ')
            for field,value in b.items():
                if value.shape!=parent[field].shape or value.dtype!=parent[field].dtype or not np.isfinite(value).all():
                    raise ValueError('invalid bank tensor')
                if field!='K' and value.tobytes()!=parent[field].tobytes():raise ValueError('non-key change')
        # Freeze candidate bytes before any ARC scoring.
        save(out/'frozen_candidates.json',{name:None if b is None else {k:__import__('hashlib').sha256(v.tobytes()).hexdigest() for k,v in b.items()} for name,b in banks.items()})
        summary={}
        for split,path in config['datasets'].items():
            rows=[json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
            if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate IDs')
            # Predeclared lexical diagnostic slice; not a semantic applicability label.
            heat={r['id'] for r in rows if re.search(r'\b(conduction|convection|radiation|evaporation|heat transfer|thermal energy)\b',r['question']+' '+' '.join(r['choices']['text']),re.I)}
            save(out/(split+'_slice_ids.json'),sorted(heat));base=None;summary[split]={}
            for name,bank in banks.items():
                state.update(status='evaluating',split=split,candidate=name,completed=0,total=len(rows));check()
                records=[];mode='off' if bank is None else 'forced' if name=='forced' else 'retrieval'
                with (out/(split+'_'+name+'.jsonl')).open('w',encoding='utf-8') as f:
                    for row in rows:
                        result=score_boundary(study,row,bank,mode);records.append(result)
                        f.write(json.dumps(result)+'\n');f.flush()
                        state['completed']=len(records);check()
                if base is None:base=records
                summary[split][name]=dict(all=summarize(records,base),heat_slice=summarize([r for r in records if r['id'] in heat],base))
                save(out/'summary.json',summary)
        if model_hash(study.model)!=before:raise ValueError('base changed')
        resources();state.update(status='complete',summary=summary,model_sha256=before);check();save(out/'results.json',state)
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}');status(out/'status.json',state);raise
