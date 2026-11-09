"""Frozen ARC science transfer/retention audit of all capability6 artifacts."""
from contextlib import nullcontext
import json
from pathlib import Path
import time
from skillforge.arc_eval import model_hash, score_row, validate_row
from skillforge.capability_bridge import modules, attached
from skillforge.go_replay import sha
from skillforge.reliable_json import save, status


def paired(before, after, field='correct'):
    if len(before) != len(after) or not before:
        raise ValueError('nonempty paired rows required')
    if any(a['id'] != b['id'] or a['gold'] != b['gold'] for a,b in zip(before,after)):
        raise ValueError('paired identity differs')
    corrections=sum(not a[field] and b[field] for a,b in zip(before,after))
    regressions=sum(a[field] and not b[field] for a,b in zip(before,after))
    n=len(before)
    return dict(n=n, corrections=corrections, regressions=regressions,
                net_correct=corrections-regressions, accuracy_delta=(corrections-regressions)/n)


def load_rows(path, expected):
    rows=[json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
    if len(rows)!=expected or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('dataset count/identity')
    for row in rows:validate_row(row)
    return rows


def restore_foundation(study, config):
    import torch
    if sha(config['checkpoint_path'])!=config['checkpoint_sha256']:
        raise ValueError('foundation checkpoint changed')
    state=torch.load(config['checkpoint_path'],map_location='cpu',weights_only=False)
    study.model.load_state_dict(state['model'],strict=True)
    study.model.eval().requires_grad_(False)
    if model_hash(study.model)!=config['model_sha256']:
        raise ValueError('foundation model identity')


def run(constructor, output_dir, config):
    import torch
    out=Path(output_dir); started=time.monotonic()
    receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:
        raise ValueError('approved source required')
    state=dict(status='verifying',manifest_sha256=receipt['manifest_sha256'],summaries={},comparisons={})
    def snapshot():
        state['active_seconds']=time.monotonic()-started
        status(out/'status.json',state)
    def resources():
        for path,h in config['resources'].items():
            if sha(path)!=h:raise ValueError('resource changed: '+path)
    def guard():
        control=out/'control.json'
        if time.monotonic()-started>config['wall_guard_seconds'] or (control.exists() and json.loads(control.read_text()).get('stop')):
            raise RuntimeError('stopped before completion')
    try:
        snapshot();resources()
        rows={k:load_rows(v['path'],v['n']) for k,v in config['datasets'].items()}
        state['status']='loading';snapshot();study=constructor()
        original=model_hash(study.model)
        if original!=config['original_model_sha256']:raise ValueError('original model identity')
        records={};conditions=['original','foundation']+list(config['adapters'])
        state['total_rows']=sum(map(len,rows.values()))*len(conditions);state['completed_rows']=0
        for condition in conditions:
            guard()
            if condition=='foundation':restore_foundation(study,config['foundation'])
            base_expected=original if condition=='original' else config['foundation']['model_sha256']
            bank=None
            if condition in config['adapters']:
                plan=json.loads(Path(config['adapters'][condition]).read_text())
                bank=modules(plan).eval().requires_grad_(False)
            bank_before=None if bank is None else model_hash(bank)
            records[condition]={};state['summaries'][condition]={}
            with nullcontext() if bank is None else attached(study.model,bank):
                for split,data in rows.items():
                    scores=[]
                    with (out/(condition+'_'+split+'.jsonl')).open('x',encoding='utf-8') as log:
                        for row in data:
                            guard();score=score_row(study.model,study.tokenizer,row)
                            scores.append(score);log.write(json.dumps(score,allow_nan=False)+'\n');log.flush()
                            state.update(status='running',condition=condition,split=split,split_completed=len(scores),split_total=len(data))
                            state['completed_rows']+=1;snapshot()
                    records[condition][split]=scores
                    state['summaries'][condition][split]=dict(n=len(scores),correct=sum(s['correct'] for s in scores),
                        accuracy=sum(s['correct'] for s in scores)/len(scores),raw_correct=sum(s['raw_correct'] for s in scores),
                        raw_accuracy=sum(s['raw_correct'] for s in scores)/len(scores))
            if model_hash(study.model)!=base_expected or any(p.grad is not None for p in study.model.parameters()):
                raise ValueError('base mutated')
            if bank is not None and model_hash(bank)!=bank_before:raise ValueError('adapter mutated')
            state['comparisons'][condition]={ref:{split:{field:paired(records[ref][split],records[condition][split],field)
                for field in ('correct','raw_correct')} for split in rows}
                for ref in ('original','foundation') if ref in records}
            del bank;torch.cuda.empty_cache();snapshot()
        resources()
        state.update(status='complete',peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            interpretation='Previously used ARC science test splits; descriptive transfer/retention, no tuning or fresh confirmation.',
            next_step='No automatic follow-on.')
    except BaseException as error:
        state.update(status='incomplete',error=f'{type(error).__name__}: {error}');raise
    finally:
        snapshot();save(out/'results.json',state)
