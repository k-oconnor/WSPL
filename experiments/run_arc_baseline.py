"""Baseline feasibility only; validation splits, no author calls or fitting."""
import json
from pathlib import Path
import time
from skillforge.arc_eval import score_row,model_hash
from skillforge.go_replay import sha
from run_diffusion_replay import save


def run(constructor,output_dir,config):
    import torch
    out=Path(output_dir);receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('approved source required')
    for path,h in config['resources'].items():
        if sha(path)!=h:raise ValueError('resource changed: '+path)
    started=time.monotonic();state=dict(status='loading',manifest_sha256=receipt['manifest_sha256']);save(out/'status.json',state)
    try:
        study=constructor();before=model_hash(study.model);summaries={}
        for task,path in config['datasets'].items():
            rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()];scores=[]
            with (out/(task+'.jsonl')).open('w',encoding='utf-8') as log:
                for row in rows:
                    control=out/'control.json'
                    if time.monotonic()-started>config['wall_guard_seconds'] or (control.exists() and json.loads(control.read_text()).get('stop')):raise RuntimeError('stopped before completion')
                    score=score_row(study.model,study.tokenizer,row);scores.append(score);log.write(json.dumps(score)+'\n');log.flush()
                    state.update(status='running',task=task,completed=len(scores),total=len(rows),active_seconds=time.monotonic()-started);save(out/'status.json',state)
            summaries[task]=dict(n=len(scores),accuracy=sum(r['correct'] for r in scores)/len(scores),raw_accuracy=sum(r['raw_correct'] for r in scores)/len(scores),chance=sum(r['chance'] for r in scores)/len(scores))
        after=model_hash(study.model)
        if after!=before:raise ValueError('model changed during evaluation')
        state.update(status='complete',summaries=summaries,unchanged_model_sha256=before,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            suitable_for_authoring=summaries['challenge']['chance']+.03<=summaries['challenge']['accuracy']<=.75,
            suitability_note='Descriptive gate only: above chance by at least 3pp and below 75%; no automatic authoring.')
        for path,h in config['resources'].items():
            if sha(path)!=h:raise ValueError('resource changed after evaluation: '+path)
    except BaseException as e:
        state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:
        state['active_seconds']=time.monotonic()-started;save(out/'status.json',state);save(out/'results.json',state)
