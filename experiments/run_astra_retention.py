"""Retention-constrained neural selector engineering; frozen controllers/executives."""
import gzip
import json
import time
from pathlib import Path
import torch
from skillforge.adaptation_pilot import episode,tasks
from skillforge.retention_pilot import DEFAULT_SOURCE,compile_selector,diagnose,parse_request,packet
from skillforge.wspl_artifact import ReLUArtifact
from skillforge.authoring_sft import restore_adapter,frozen_hash
from skillforge.closed_loop_control import prompt,teacher
from skillforge.executive_study import load_library
from skillforge.codex_one_shot import collect
from skillforge.reliable_json import save,status
from run_closed_loop import file_hash


def run(constructor,output_dir,config):
    out=Path(output_dir);start=time.monotonic();live={}
    def report(**kw):
        live.update(kw,elapsed_seconds=time.monotonic()-start);status(out/'status.json',live)
    def stopped():
        p=out/'control.json'
        return time.monotonic()-start>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def guard():
        if stopped():raise InterruptedError('Stop or study guard')
    try:
        report(status='verifying')
        for path,h in config['resources'].items():
            guard()
            if file_hash(path)!=h:raise ValueError('Changed resource '+path)
        blocks,specs=load_library(config['library']);repair=ReLUArtifact.load(config['repair'])
        library_ids={k:v.sha256 for k,v in blocks.items()};repair_id=repair.sha256
        study=constructor();model,tokenizer=study.model,study.tokenizer
        model.eval();model.gradient_checkpointing_disable();base=frozen_hash(model)
        def load(seed):
            saved=torch.load(config['executives'][seed],map_location='cpu',weights_only=False)
            if saved['base_hash']!=base or saved['manifest_sha256']!=config['executive_manifest']:raise ValueError('Executive identity')
            restore_adapter(model,saved['adapter'])
        def action(observation,label):
            guard();messages=[dict(role='user',content=prompt(observation))]
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            ids=tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
            if ids.shape[1]+64>8192:raise ValueError('Context overflow')
            with torch.inference_mode():
                output=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,use_cache=True,max_new_tokens=64,pad_token_id=tokenizer.eos_token_id)
            tokens=output[0,ids.shape[1]:].tolist();answer=tokenizer.decode(tokens,skip_special_tokens=True)
            with (out/'local_replies.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(label=label,prompt=messages,answer=answer,tokens=len(tokens)))+'\n')
            return answer
        selector=compile_selector(DEFAULT_SOURCE);command=[0.,0.];source=DEFAULT_SOURCE
        identity=[[1.,0.],[0.,1.]];changed=config['mapping']
        calibration=tasks(21000000,[3,3,5,5])
        reserved=tasks(22000000,[3]*4+[5]*4+[8]*4+[12]*4)
        # Same installed artifact; episode reset changes physical scene, not weights.
        phases=[('original_before',identity),('changed',changed),('original_return',identity)]
        save(out/'partitions.json',dict(calibration=calibration,reserved=reserved,phase_order=[p for p,_ in phases]))
        def execute(task,mapping,label,arm,reference=False):
            e=episode(task,blocks,specs,mapping);diagnostic=None
            if arm!='baseline':
                selected,diagnostic=diagnose(e.world,command,selector);e.ticks+=1
                if not e.world.safe():e.finished=True;e.error='diagnostic_unsafe';e.violations.append('unsafe')
                if arm=='matched_original':selected='original'
                if arm=='reference':selected='original' if mapping==identity else 'repaired'
                e.blocks=dict(blocks,move=repair if selected=='repaired' else blocks['move'])
            while not e.finished:
                guard();e.execute(teacher(e.observation()) if reference else action(e.observation(),label))
            r=e.result();trace=r.pop('trace');decisions=r.pop('decision_trace')
            path=label+'.trace.json.gz'
            (out/path).write_bytes(gzip.compress(json.dumps(dict(diagnostic=diagnostic,trace=trace,decisions=decisions),separators=(',',':')).encode(),mtime=0))
            r.update(trace_path=path,diagnostic=diagnostic);return r
        def acquisition(label):
            load('113');rows=[]
            for regime,mapping in enumerate((identity,changed)):
                for i,t in enumerate(calibration):
                    report(status='calibration',label=label,case=regime*4+i)
                    r=execute(t,mapping,f'{label}_{regime}_{i}','selected')
                    trace=json.loads(gzip.decompress((out/r['trace_path']).read_bytes()))
                    rows.append(dict(case=regime*4+i,success=r['task_success'],violations=r['violations'],error=r['error'],
                        diagnostic=r['diagnostic'],decisions=trace['decisions'][:3]))
            save(out/(label+'.json'),rows);return rows
        history=[dict(initial_calibration=acquisition('initial'))];remaining=dict(responses=8,evaluations=4)
        for turn in range(1,9):
            remaining['responses']=9-turn;report(status='astra',turn=turn)
            def heartbeat():report(status='astra',turn=turn);return stopped()
            answer,audit=collect(config['cli'],config['catalog'],config['instructions'],out/f'turn_{turn}',
                                 packet(blocks['move'],repair,history,remaining),1200,heartbeat)
            entry=dict(assistant=answer,usage=audit['usage']);finish=False
            try:
                a=parse_request(answer)
                if a['tool']=='install':
                    proposed=compile_selector(a['source']);selector=proposed;source=a['source'];command=a['command']
                    feedback=dict(installed=True,selector=selector.to_dict(),command=command)
                elif a['tool']=='evaluate':
                    if remaining['evaluations']<=0:raise ValueError('Evaluation budget exhausted')
                    remaining['evaluations']-=1;feedback=dict(episodes=acquisition(f'calibration_{turn}'))
                else:finish=True;feedback=dict(frozen=True)
                entry['feedback']=dict(accepted=True,result=feedback)
            except (ValueError,KeyError,TypeError) as err:entry['feedback']=dict(accepted=False,error=str(err))
            history.append(entry);save(out/'history.json',history)
            save(out/'selector_last.json',dict(weights=selector.to_dict(),command=command,source=source))
            if finish:break
        final=dict(weights=selector.to_dict(),command=command,source=source)
        save(out/'selector_final.json',final)
        frozen=dict(selector_file=file_hash(out/'selector_final.json'),executives={s:file_hash(p) for s,p in config['executives'].items()},
                    library=library_ids,repair=repair_id)
        save(out/'frozen.json',frozen);outcomes=[]
        for seed in config['executives']:
            load(seed)
            for arm in ('baseline','matched_original','selected'):
                for phase,mapping in phases:
                    for i,t in enumerate(reserved):
                        report(status='final_evaluation',seed=seed,arm=arm,phase=phase,completed=i,total=len(reserved))
                        r=execute(t,mapping,f'{seed}_{arm}_{phase}_{i}',arm)
                        row=dict(seed=seed,arm=arm,phase=phase,index=i,**r);outcomes.append(row)
                        with (out/'scores.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        for phase,mapping in phases:
            for i,t in enumerate(reserved):
                outcomes.append(dict(seed='reference',arm='reference',phase=phase,index=i,
                    **execute(t,mapping,f'reference_{phase}_{i}','reference',True)))
        summary={};paired={}
        for row in outcomes:
            k='/'.join((row['seed'],row['arm'],row['phase']));v=summary.setdefault(k,dict(n=0,success=0))
            v['n']+=1;v['success']+=row['task_success']
        for seed in config['executives']:
            for phase,_ in phases:
                a={r['index']:r['task_success'] for r in outcomes if r['seed']==seed and r['arm']=='baseline' and r['phase']==phase}
                b={r['index']:r['task_success'] for r in outcomes if r['seed']==seed and r['arm']=='selected' and r['phase']==phase}
                paired[seed+'/'+phase]=dict(n=len(a),improvements=sum(b[i] and not a[i] for i in a),
                    regressions=sum(a[i] and not b[i] for i in a),gain=(sum(b.values())-sum(a.values()))/len(a))
        ref_ok=all(r['task_success'] for r in outcomes if r['arm']=='reference')
        selections={}
        for row in outcomes:
            if row['arm']!='selected':continue
            key=row['seed']+'/'+row['phase'];v=selections.setdefault(key,dict(n=0,correct=0,diagnostic_steps=0))
            wanted='repaired' if row['phase']=='changed' else 'original'
            v['n']+=1;v['correct']+=row['diagnostic']['selected']==wanted
            v['diagnostic_steps']+=row['diagnostic']['physical_steps']
        retention=all(paired[s+'/'+p]['regressions']==0 for s in config['executives'] for p in ('original_before','original_return'))
        improved=all(paired[s+'/changed']['gain']>=.2 for s in config['executives'])
        assert frozen_hash(model)==base and library_ids=={k:v.sha256 for k,v in blocks.items()} and repair.sha256==repair_id
        assert file_hash(out/'selector_final.json')==frozen['selector_file']
        assert all(file_hash(config['executives'][s])==h for s,h in frozen['executives'].items())
        save(out/'scores.json',outcomes)
        result=dict(status='complete',summary=summary,paired=paired,selections=selections,reference_passed=ref_ok,
          retention_passed=retention,improvement_passed=improved,feasibility_passed=ref_ok and retention and improved,
          author_responses=len(history)-1,artifact_frozen=True,executives_and_blocks_unchanged=True,
          elapsed_seconds=time.monotonic()-start)
        save(out/'results.json',result);report(status='complete',results=result)
    except BaseException as err:
        report(status='paused' if isinstance(err,InterruptedError) else 'failed',error=repr(err));raise
