"""Source-gated learned executive pilot. No direct entry point or auto-retry."""
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from skillforge.authoring_sft import adapter_state, restore_adapter, frozen_hash, encode, loss_sum, checkpoint
from skillforge.executive_study import dataset, load_library, score
from skillforge.reliable_json import save, status
from skillforge import unified_executive
from skillforge.wspl_language import library_document


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def run(constructor, output_dir, config):
    out = Path(output_dir)
    started = time.monotonic()
    live = dict(status='verifying', step=0)
    model = optimizer = None
    step = targets = 0
    updating = finished_training = False
    manifest = json.loads((out/'source_receipt.json').read_text())['manifest_sha256']

    def report(**values):
        live.update(values, elapsed_seconds=time.monotonic()-started)
        status(out/'status.json',live)

    def guard():
        if time.monotonic()-started>config['wall_guard_seconds']:
            raise InterruptedError('Study wall guard')
        p=out/'control.json'
        if p.exists() and json.loads(p.read_text()).get('stop'):
            raise InterruptedError('User stop')

    def preserve():
        checkpoint(out/'rolling.pt',model,optimizer,dict(step=step,targets=targets,
                   manifest_sha256=manifest,base_hash=base_hash,config=config))

    try:
        report()
        for path,h in config['resources'].items():
            if digest(path)!=h:
                raise ValueError('Resource changed: '+path)
        blocks,specs=load_library(config['library'])
        unified=config.get('language')=='wspl0.3'
        library=library_document(blocks,specs)
        data=unified_executive.dataset(library) if unified else dataset()
        reused=unified_executive.reused_replies(config,data['test']) if unified else []
        save(out/'linked_library.json',library)
        for split,rows in data.items():
            with (out/(split+'.jsonl')).open('x',encoding='utf8') as f:
                for row in rows:
                    f.write(json.dumps(row)+'\n')
        save(out/'corpus_identity.json',{k:dict(rows=len(v),sha256=digest(out/(k+'.jsonl'))) for k,v in data.items()})
        study=constructor()
        model,tokenizer=study.model,study.tokenizer
        base_hash=frozen_hash(model)
        initial=adapter_state(model)
        train=[encode(tokenizer,r['messages'],config['context']) for r in data['train']]
        dev=[encode(tokenizer,r['messages'],config['context']) for r in data['development']]
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=.01,foreach=False)
        validation=[]

        def validate():
            model.eval()
            optimizer.zero_grad(set_to_none=True)
            gc.collect();torch.cuda.empty_cache()
            total=count=0
            with torch.no_grad():
                for i,r in enumerate(dev):
                    guard()
                    total+=float(loss_sum(model,r));count+=r['targets']
                    if i%16==0:
                        report(status='validation',step=step,completed=i,total=len(dev))
            validation.append(dict(step=step,nll=total/count))
            save(out/'validation.json',validation)
            model.train();gc.collect();torch.cuda.empty_cache()

        validate()
        total_steps=config['epochs']*math.ceil(len(train)/config['accumulation'])
        last_save=time.monotonic()
        with (out/'training.jsonl').open('x',encoding='utf8') as log:
            for epoch in range(config['epochs']):
                order=list(range(len(train)));random.Random(711+epoch).shuffle(order)
                for offset in range(0,len(order),config['accumulation']):
                    guard()
                    records=[train[i] for i in order[offset:offset+config['accumulation']]]
                    count=sum(r['targets'] for r in records);loss_total=0.
                    optimizer.zero_grad(set_to_none=True)
                    for record in records:
                        loss=loss_sum(model,record)
                        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                        (loss/count).backward();loss_total+=float(loss.detach());del loss
                    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                    fraction=max(0.,(step+1-config['warmup'])/(total_steps-config['warmup']))
                    lr=config['lr']*min(1.,(step+1)/config['warmup'])*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
                    for g in optimizer.param_groups:g['lr']=lr
                    updating=True;optimizer.step();updating=False
                    optimizer.zero_grad(set_to_none=True);step+=1;targets+=count
                    report(status='training',step=step,total_steps=total_steps,epoch=epoch+1,nll=loss_total/count,grad_norm=float(norm),lr=lr,targets=targets)
                    log.write(json.dumps(live)+'\n');log.flush()
                    if time.monotonic()-last_save>=1800:
                        preserve();last_save=time.monotonic()
                preserve();last_save=time.monotonic();validate()
        if frozen_hash(model)!=base_hash:raise ValueError('Frozen base changed')
        preserve();final=adapter_state(model)
        torch.save(dict(adapter=final,manifest_sha256=manifest,base_hash=base_hash,step=step),out/'adapter_final.pt')
        finished_training=True
        model.gradient_checkpointing_disable();model.eval()
        replies=[]
        if unified:
            # Same task identities and frozen old outputs; no redundant model generation.
            replies=reused
            with (out/'replies.jsonl').open('x',encoding='utf8') as f:
                for r in replies:f.write(json.dumps(r)+'\n')
            arms=[('trained',final),('base',initial)]
        else:
            prior=torch.load(config['prior_adapter'],map_location='cpu',weights_only=False)
            if prior['base_hash']!=base_hash:raise ValueError('Previous executive base identity differs')
            if prior['manifest_sha256']!=config['prior_manifest_sha256']:raise ValueError('Previous executive source identity differs')
            arms=[('trained',final),('previous',prior['adapter']),('base',initial)]
        with torch.inference_mode():
            for arm,weights in arms:
                restore_adapter(model,weights)
                for i,row in enumerate(data['test']):
                    guard();report(status='generation',arm=arm,completed=i,total=len(data['test']))
                    text=tokenizer.apply_chat_template(row['messages'][:1],tokenize=False,add_generation_prompt=True)
                    ids=tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
                    begin=time.monotonic()
                    output=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,use_cache=True,
                        max_new_tokens=config.get('generation_tokens',512),pad_token_id=tokenizer.eos_token_id)
                    tokens=output[0,ids.shape[1]:].tolist()
                    reply=dict(arm=arm,id=row['id'],source=tokenizer.decode(tokens,skip_special_tokens=True),
                               tokens=tokens,seconds=time.monotonic()-begin,hit_cap=len(tokens)==config.get('generation_tokens',512))
                    replies.append(reply)
                    with (out/'replies.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(reply)+'\n')
        # Freeze all learned-model replies before any execution feedback.
        tasks={r['id']:r for r in data['test']}
        candidates=replies+[dict(arm='reference',id=r['id'],source=r['messages'][1]['content']) for r in data['test']]
        scores=[]
        for i,candidate in enumerate(candidates):
            guard();report(status='execution',completed=i,total=len(candidates))
            scorer=unified_executive.score if unified and candidate['arm']!='previous' else score
            result=scorer(candidate['source'],tasks[candidate['id']]['task'],blocks,specs)
            trace=result.pop('trace',[])
            save(out/(candidate['arm']+'_'+candidate['id']+'.trace.json'),trace)
            result.update(arm=candidate['arm'],id=candidate['id']);scores.append(result)
        save(out/'scores.json',scores)
        if unified:
            historical_scores=json.loads(Path(config['previous_scores']).read_text())
            expected={r['id']:r for r in historical_scores if r['arm']=='trained'}
            for row in scores:
                if row['arm']=='previous':
                    old=expected[row['id']]
                    if {k:v for k,v in row.items() if k!='arm'}!={k:v for k,v in old.items() if k!='arm'}:
                        raise ValueError('Historical execution replay differs')
        if frozen_hash(model)!=base_hash:raise ValueError('Frozen base changed at evaluation')
        summary={}
        for arm in ['base','previous','trained','reference']:
            summary[arm]={}
            for panel in sorted({r['panel'] for r in scores}):
                rows=[r for r in scores if r['arm']==arm and r['panel']==panel]
                summary[arm][panel]=dict(total=len(rows),constructed=sum(r['constructed'] for r in rows),
                    task_success=sum(r['task_success'] for r in rows),safe_failure=sum(r['safe_failure'] for r in rows))
        paired={}
        lookup={(r['arm'],r['id']):r for r in scores}
        for panel in sorted({r['panel'] for r in scores}):
            metric='safe_failure' if panel.startswith('jammed') else 'task_success'
            rows=[r for r in scores if r['arm']=='trained' and r['panel']==panel]
            paired[panel]=dict(metric=metric,improvements=sum(r[metric] and not lookup['previous',r['id']][metric] for r in rows),
                regressions=sum(not r[metric] and lookup['previous',r['id']][metric] for r in rows),
                unchanged=sum(r[metric]==lookup['previous',r['id']][metric] for r in rows))
        result=dict(status='complete',manifest_sha256=manifest,base_hash=base_hash,validation=validation,paired_vs_previous=paired,
            summary=summary,reference_gate_passed=all(r['safe_failure'] if r['panel'].startswith('jammed') else r['task_success'] for r in scores if r['arm']=='reference'),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            language=config.get('language','legacy'),historical_replay_exact=unified,
            usage={a:dict(tokens=sum(len(r['tokens']) for r in replies if r['arm']==a),seconds=sum(r['seconds'] for r in replies if r['arm']==a),reused=unified and a=='previous') for a in ['base','previous','trained']})
        save(out/'results.json',result);report(status='complete',results=result)
    except BaseException as error:
        if model is not None and optimizer is not None and not updating and not finished_training:
            optimizer.zero_grad(set_to_none=True);preserve()
        report(status='paused' if isinstance(error,InterruptedError) else 'failed',error=repr(error))
        raise
