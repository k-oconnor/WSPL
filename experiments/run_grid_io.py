"""Frozen grid interface prerequisite; no adapter fits or authoring calls."""
import json,time
from pathlib import Path
from skillforge.grid_io import generate,summarize
from skillforge.arc_eval import model_hash
from skillforge.go_replay import sha
from skillforge.reliable_json import save,status


def run(constructor,output_dir,config):
    out=Path(output_dir);receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('receipt')
    def resources():
        for p,h in config['resources'].items():
            if sha(p)!=h:raise ValueError('resource changed: '+p)
    resources();start=time.monotonic();state=dict(status='loading',manifest_sha256=receipt['manifest_sha256'])
    def snapshot():state['active_seconds']=time.monotonic()-start;status(out/'status.json',state)
    def check():
        p=out/'control.json'
        if time.monotonic()-start>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop')):raise RuntimeError('stop/guard')
    try:
        study=constructor();before=model_hash(study.model)
        if before!=config['model_sha256']:raise ValueError('model identity')
        data=json.loads(Path(config['data']).read_text());demos=data['demonstrations'];summary={}
        for name in ('copy','movement_familiar','movement_novel'):
            task='copy' if name=='copy' else 'movement';summary[name]={}
            for mode in ('unconstrained','palette_only'):
                records=[];state.update(status='evaluating',split=name,mode=mode,completed=0,total=len(data[name]));snapshot()
                with (out/(name+'_'+mode+'.jsonl')).open('w',encoding='utf-8') as f:
                    for row in data[name]:
                        check();answer=generate(study,row['input'],task,demos,mode)
                        target=row['input'] if task=='copy' else row['output']
                        exact=answer['output']==target
                        cells=sum(a==b for a,b in zip(answer['output'],target)) if answer['output'] is not None else 0
                        record=dict(answer,id=row['id'],shape=row['shape'],direction=row['direction'],color=row['color'],exact=exact,correct_cells=cells)
                        records.append(record);f.write(json.dumps(record)+'\n');f.flush();state['completed']=len(records);snapshot()
                summary[name][mode]=summarize(records);save(out/'summary.json',summary)
        if model_hash(study.model)!=before:raise ValueError('model changed')
        resources();copy=summary['copy']['palette_only'];raw=summary['copy']['unconstrained']
        state.update(status='complete',summary=summary,model_sha256=before,
            prerequisite=dict(copy_gate_passed=copy['exact']>=61,threshold='61/64 exact grids with palette-only format constraint',
                              unconstrained_copy_passed=raw['exact']>=61,
                              next_step='Prepare gradient-trained ReLU adapter reachability control only if copy gate passes; otherwise resolve representation/interface first. No automatic follow-on.'))
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
