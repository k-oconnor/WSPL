"""Source-gated compact closed-loop study; no direct launcher or automatic retry."""
import copy
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import torch

from skillforge.authoring_sft import (LoRALinear,adapter_state,restore_adapter,frozen_hash,
                                     encode,loss_sum,checkpoint)
from skillforge.closed_loop_control import Episode,training_data,evaluation_tasks,prompt,teacher
from skillforge.controller_author import construct,teacher as block_teacher
from skillforge.executive_study import load_library,prompt as old_prompt
from skillforge.local_mechanism import calibration,author_prompt,fit_replacement
from skillforge.wspl_parse import parse_module
from skillforge.reliable_json import save,status


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def reset_adapter(model,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module,LoRALinear):
                torch.nn.init.kaiming_uniform_(module.lora_A,a=math.sqrt(5));module.lora_B.zero_()


def fit_seed(model,tokenizer,data,out,config,seed,manifest,base_hash,guard,report):
    out.mkdir();reset_adapter(model,seed)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train();train=[encode(tokenizer,r['messages'],2048) for r in data['train']]
    dev=[encode(tokenizer,r['messages'],2048) for r in data['development']]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002,weight_decay=.01,foreach=False)
    step=targets=0;updating=False;validation=[];last_save=time.monotonic()
    phase_start=time.monotonic();training_seconds=validation_seconds=0.
    total_steps=config['epochs']*math.ceil(len(train)/8)

    def preserve():
        checkpoint(out/'rolling.pt',model,optimizer,dict(seed=seed,step=step,targets=targets,
                   manifest_sha256=manifest,base_hash=base_hash))

    def validate():
        nonlocal validation_seconds
        begin=time.monotonic()
        model.eval();optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        total=count=0
        with torch.no_grad():
            for i,row in enumerate(dev):
                guard();total+=float(loss_sum(model,row));count+=row['targets']
                if i%32==0:report(status='validation',seed=seed,step=step,completed=i,total=len(dev))
        validation.append(dict(step=step,nll=total/count));save(out/'validation.json',validation)
        model.train();gc.collect();torch.cuda.empty_cache()
        validation_seconds+=time.monotonic()-begin

    try:
        validate()
        with (out/'training.jsonl').open('x',encoding='utf8') as f:
            for epoch in range(config['epochs']):
                order=list(range(len(train)));random.Random(seed+epoch+711).shuffle(order)
                for offset in range(0,len(order),8):
                    guard();optimizer.zero_grad(set_to_none=True)
                    update_start=time.monotonic()
                    rows=[train[i] for i in order[offset:offset+8]];count=sum(r['targets'] for r in rows);value=0.
                    for row in rows:
                        loss=loss_sum(model,row)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                        (loss/count).backward();value+=float(loss.detach());del loss
                    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
                    fraction=max(0.,(step+1-160)/(total_steps-160))
                    lr=.0002*min(1.,(step+1)/160)*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
                    for group in optimizer.param_groups:group['lr']=lr
                    updating=True;optimizer.step();updating=False;optimizer.zero_grad(set_to_none=True)
                    step+=1;targets+=count
                    record=dict(seed=seed,step=step,total_steps=total_steps,nll=value/count,grad_norm=float(norm),lr=lr,targets=targets)
                    torch.cuda.synchronize();duration=time.monotonic()-update_start;training_seconds+=duration
                    record.update(update_seconds=duration,training_seconds=training_seconds,elapsed_seconds=time.monotonic()-phase_start)
                    report(status='training',**record);f.write(json.dumps(record)+'\n');f.flush()
                    if time.monotonic()-last_save>=1800:preserve();last_save=time.monotonic()
                preserve();last_save=time.monotonic();validate()
        if frozen_hash(model)!=base_hash:raise ValueError('Frozen base changed')
        preserve();torch.save(dict(adapter=adapter_state(model),base_hash=base_hash,manifest_sha256=manifest,
                                  seed=seed,step=step),out/'adapter_final.pt')
        save(out/'training_summary.json',dict(seed=seed,steps=step,targets=targets,training_seconds=training_seconds,
                                             validation_seconds=validation_seconds,elapsed_seconds=time.monotonic()-phase_start))
    except BaseException:
        if not updating:optimizer.zero_grad(set_to_none=True);preserve()
        raise
    finally:
        del optimizer;gc.collect();torch.cuda.empty_cache()


def run(constructor,output_dir,config):
    out=Path(output_dir);start=time.monotonic();live={}
    manifest=json.loads((out/'source_receipt.json').read_text())['manifest_sha256']

    def report(**values):
        live.update(values,elapsed_seconds=time.monotonic()-start);status(out/'status.json',live)

    def guard():
        if time.monotonic()-start>config['wall_guard_seconds']:raise InterruptedError('Wall guard')
        p=out/'control.json'
        if p.exists() and json.loads(p.read_text()).get('stop'):raise InterruptedError('User stop')

    try:
        report(status='verifying')
        for path,h in config['resources'].items():
            guard()
            if file_hash(path)!=h:raise ValueError('Resource changed: '+path)
        blocks,specs=load_library(config['library'])
        original_ids={k:v.sha256 for k,v in blocks.items()}
        report(status='building_training_snapshots')
        data=training_data(blocks,specs,config['train_rows'],config['dev_rows'],guard,report)
        tasks=evaluation_tasks()
        for name,rows in dict(data,test=tasks).items():
            with (out/(name+'.jsonl')).open('x',encoding='utf8') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
        save(out/'corpus_identity.json',{name:file_hash(out/(name+'.jsonl')) for name in ('train','development','test')})
        guard();study=constructor();model,tokenizer=study.model,study.tokenizer
        base_hash=frozen_hash(model);untouched=adapter_state(model)
        for seed in config['seeds']:
            fit_seed(model,tokenizer,data,out/f'seed_{seed}',config,seed,manifest,base_hash,guard,report)
        model.gradient_checkpointing_disable();model.eval()

        def generate(messages,cap,label):
            guard();report(status='generation',label=label)
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            ids=tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
            if ids.shape[1]+cap>8192:raise ValueError('Generation context budget exceeded')
            begin=time.monotonic()
            with torch.inference_mode():
                output=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,use_cache=True,
                                      max_new_tokens=cap,pad_token_id=tokenizer.eos_token_id)
            tokens=output[0,ids.shape[1]:].tolist()
            row=dict(label=label,prompt=messages,source=tokenizer.decode(tokens,skip_special_tokens=True),
                     tokens=tokens,prompt_tokens=int(ids.shape[1]),seconds=time.monotonic()-begin,hit_cap=len(tokens)==cap)
            with (out/'replies.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
            return row['source']

        def load_old(key,expected):
            saved=torch.load(config[key],map_location='cpu',weights_only=False)
            if saved['base_hash']!=base_hash or saved['manifest_sha256']!=expected:
                raise ValueError('Historical adapter identity differs')
            restore_adapter(model,saved['adapter'])

        # All executive checkpoints are frozen before authoring/fitting replacements.
        replacement_spec=dict(specs['move'],gain=-specs['move']['gain'])
        rows=calibration(replacement_spec);save(out/'replacement_calibration.json',rows)
        load_old('author_adapter','5de7ada4ce5d2c3c41614896e568cca4b91bc63991f5f4ce997a7e9404d1f3ae')
        authored_source=generate([dict(role='user',content=author_prompt(replacement_spec,rows))],512,'movement_author')
        (out/'movement_authored.wspl').write_text(authored_source,encoding='utf8')
        try:
            authored=construct(authored_source,replacement_spec)
            if authored.to_dict()['allocation']['parameters']>blocks['move'].to_dict()['allocation']['parameters']:
                raise ValueError('Authored replacement exceeds matched parameter budget')
            authored.save(out/'movement_authored.weights.json')
            author_construction=dict(valid=True,sha256=authored.sha256)
        except (ValueError,TypeError,KeyError) as error:
            authored=None;author_construction=dict(valid=False,error=str(error))
        reference=construct(block_teacher(replacement_spec),replacement_spec)
        reference.save(out/'movement_reference.weights.json')
        source_ir=parse_module(block_teacher(replacement_spec)).program
        save(out/'movement_reference.ir.json',source_ir)
        gradients={};fits=[]
        for seed in config['seeds']:
            guard();report(status='replacement_fitting',seed=seed)
            artifact,metrics=fit_replacement(blocks['move'],rows,seed,config['replacement_updates'],guard)
            artifact.save(out/f'movement_adamw_{seed}.weights.json');gradients[seed]=artifact;fits.append(metrics)
        save(out/'replacement_construction.json',dict(author=author_construction,adamw=fits))
        # Exact real-graph equivalence is reported separately from rollout performance.
        for name,path in [('reference',out/'movement_reference.weights.json')]+(
                [('authored',out/'movement_authored.weights.json')] if authored is not None else []):
            env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
            completed=subprocess.run([config['proof_python'],'-m','skillforge.wspl_equiv',str(out/'movement_reference.ir.json'),
                                      str(path),'--specification','--output',str(out/f'{name}_equivalence.json')],env=env,
                                     capture_output=True,text=True,timeout=30)
            (out/f'{name}_equivalence.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf8')
            if not (out/f'{name}_equivalence.json').exists():raise ValueError('Proof runtime failed')
        save(out/'frozen_artifacts.json',dict(executives={str(s):file_hash(out/f'seed_{s}/adapter_final.pt') for s in config['seeds']},
                                             original_library=original_ids,reference=reference.sha256,
                                             authored=authored.sha256 if authored else None,
                                             gradient={str(s):a.sha256 for s,a in gradients.items()}))
        results=[]

        def record(arm,seed,index,task,episode,reused=False):
            full=episode.result();trace=full.pop('trace');decisions=full.pop('decision_trace')
            identity=f'{arm}_{seed}_{index}'
            trace_path=identity+'.trace.json.gz'
            payload=json.dumps(dict(trace=trace,decisions=decisions,events=full['events']),separators=(',',':'),allow_nan=False).encode()
            with (out/trace_path).open('xb') as f:f.write(gzip.compress(payload,compresslevel=6,mtime=0))
            row=dict(full,arm=arm,seed=seed,index=index,panel=task['panel'],reused=reused,trace_path=trace_path)
            results.append(row)
            with (out/'scores.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
            return row

        def execute_model(task,bank,label,scale=1):
            episode=Episode(task,bank,specs,scale)
            while not episode.finished:
                source=generate([dict(role='user',content=prompt(episode.observation()))],64,label+f':{len(episode.decisions)}')
                episode.execute(source)
            return episode

        def replay(task,bank,actions,scale=1):
            episode=Episode(task,bank,specs,scale)
            for action in actions:
                guard()
                if episode.finished:break
                episode.execute(action)
            if not episode.finished:episode.finished=True;episode.error='plan_exhausted'
            return episode

        # Supplied reference and untouched base use the same observed-state interface.
        restore_adapter(model,untouched)
        for i,task in enumerate(tasks):
            guard();episode=Episode(task,blocks,specs)
            while not episode.finished:episode.execute(teacher(episode.observation()))
            record('reference',0,i,task,episode)
            if task['panel']=='long8':
                inverse=Episode(task,dict(blocks,move=reference),specs,-1)
                while not inverse.finished:inverse.execute(teacher(inverse.observation()))
                record('reference_inverted_analytic',0,i,task,inverse)
            record('base',0,i,task,execute_model(task,blocks,f'base:{i}'))
        # Historical best compact model gets its original whole-plan prompt.
        load_old('previous_adapter','43dabbb5b31820586f368de2037e2673767c71eafa89c55833d2d1fa6b60f5fc')
        for i,task in enumerate(tasks):
            visible={k:task[k] for k in ('objects','points','goals','before','initial')}
            visible['blocked_points']=Episode(task,blocks,specs).blocked()
            text=generate([dict(role='user',content=old_prompt(visible))],512,f'previous:{i}')
            record('previous_open_loop',0,i,task,replay(task,blocks,text.strip().splitlines()+['stop']))
        for seed in config['seeds']:
            saved=torch.load(out/f'seed_{seed}/adapter_final.pt',map_location='cpu',weights_only=False)
            restore_adapter(model,saved['adapter'])
            for i,task in enumerate(tasks):
                episode=execute_model(task,blocks,f'closed:{seed}:{i}')
                record('closed',seed,i,task,episode)
                if task['disturbance'] in ('none','blocked'):
                    record('nominal_plan',seed,i,task,episode,reused=True)
                else:
                    nominal=dict(task,disturbance='blocked' if task['disturbance']=='combined' else 'none')
                    plan=execute_model(nominal,blocks,f'nominal:{seed}:{i}')
                    record('nominal_plan',seed,i,task,replay(task,blocks,[d['source'] for d in plan.decisions]))
                if task['panel']=='long8':
                    for name,replacement in [('unchanged',blocks['move']),('authored',authored),
                                             ('adamw',gradients[seed]),('analytic',reference)]:
                        bank=dict(blocks,move=replacement)
                        if replacement is None:
                            broken=Episode(task,blocks,specs,-1);broken.finished=True;broken.error='invalid_replacement'
                            record('inverted_'+name,seed,i,task,broken)
                        else:record('inverted_'+name,seed,i,task,execute_model(task,bank,f'inverted_{name}:{seed}:{i}',-1))
        save(out/'scores.json',results)
        if frozen_hash(model)!=base_hash or {k:b.sha256 for k,b in blocks.items()}!=original_ids:
            raise ValueError('Frozen foundation or original library changed')
        frozen=json.loads((out/'frozen_artifacts.json').read_text())
        if any(file_hash(out/f'seed_{s}/adapter_final.pt')!=frozen['executives'][str(s)] for s in config['seeds']):
            raise ValueError('Executive checkpoint changed during evaluation')
        summary={}
        for row in results:
            key=f"{row['arm']}/{row['seed']}/{row['panel']}"
            entry=summary.setdefault(key,dict(total=0,success=0,unsafe=0,omissions=0,all_actions_valid=0))
            entry['total']+=1;entry['success']+=row['task_success']
            entry['unsafe']+=bool(row['violations']);entry['omissions']+=bool(row['missing'])
            entry['all_actions_valid']+=bool(row['all_actions_valid'])
        checks={};paired={}
        lookup={(r['arm'],r['seed'],r['index']):r for r in results}
        for seed in config['seeds']:
            current=[r for r in results if r['arm']=='closed' and r['seed']==seed]
            def success(rows):return sum(r['task_success'] for r in rows)
            familiar=[r for r in current if r['panel'].startswith('familiar')]
            long=[r for r in current if r['panel'] in ('long8','long12','renamed12')]
            dynamic=[r for r in current if r['panel'] in ('slip8','moved8','blocked8','combined12')]
            improvements=sum(r['task_success'] and not lookup['nominal_plan',seed,r['index']]['task_success'] for r in dynamic)
            regressions=sum(not r['task_success'] and lookup['nominal_plan',seed,r['index']]['task_success'] for r in dynamic)
            normal=[r for r in current if r['panel']=='long8']
            authored_rows=[r for r in results if r['arm']=='inverted_authored' and r['seed']==seed]
            unpatched=[r for r in results if r['arm']=='inverted_unchanged' and r['seed']==seed]
            checks[str(seed)]=dict(familiar_90=success(familiar)>=.9*len(familiar),
                long_85=success(long)>=.85*len(long),
                dynamic_panels=all(success([r for r in dynamic if r['panel']==p])>=(.75 if p=='combined12' else .8)*16
                                   for p in ('slip8','moved8','blocked8','combined12')),
                feedback_gain_20pp=improvements-regressions>=.2*len(dynamic),
                normal_mechanism_baseline_80=success(normal)>=.8*len(normal),
                author_recovers_90=success(authored_rows)>=.9*success(normal),
                unpatched_drop_50pp=success(normal)-success(unpatched)>=.5*len(normal))
            paired[str(seed)]=dict(dynamic_total=len(dynamic),improvements=improvements,regressions=regressions,
                                   nominal_success=sum(lookup['nominal_plan',seed,r['index']]['task_success'] for r in dynamic),
                                   closed_success=success(dynamic))
        reference_passed=all(r['task_success'] for r in results if r['arm'].startswith('reference'))
        result=dict(status='complete',manifest_sha256=manifest,base_hash=base_hash,summary=summary,
                    reference_passed=reference_passed,
                    progression_passed=reference_passed and author_construction['valid'] and all(all(c.values()) for c in checks.values()),
                    descriptive_checks=checks,paired_feedback=paired,
                    author_construction=author_construction,original_library_unchanged=True,
                    executive_checkpoints_unchanged=True,elapsed_seconds=time.monotonic()-start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save(out/'results.json',result);report(status='complete',results=result)
    except BaseException as error:
        report(status='paused' if isinstance(error,InterruptedError) else 'failed',error=repr(error));raise
