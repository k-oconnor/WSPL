"""Frozen executive, bounded Astra engineering, and independent deployment panels."""
import copy
import gzip
import json
import time
from pathlib import Path

import torch
from skillforge.adaptation_pilot import episode,probe,candidate,parse_action,tasks,packet
from skillforge.authoring_sft import adapter_state,restore_adapter,frozen_hash
from skillforge.closed_loop_control import prompt,teacher
from skillforge.controller_author import teacher as block_teacher,construct
from skillforge.executive_study import load_library
from skillforge.local_mechanism import fit_replacement
from skillforge.codex_one_shot import collect
from skillforge.reliable_json import save,status
from run_closed_loop import file_hash


def run(constructor,output_dir,config):
    out=Path(output_dir);started=time.monotonic();live={}
    def report(**kw):
        live.update(kw,elapsed_seconds=time.monotonic()-started);status(out/'status.json',live)
    def stopped():
        c=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (c.exists() and json.loads(c.read_text()).get('stop'))
    def guard():
        if stopped():raise InterruptedError('Study guard or user stop')
    try:
        report(status='verifying')
        for path,h in config['resources'].items():
            guard()
            if file_hash(path)!=h:raise ValueError('Resource changed: '+path)
        blocks,specs=load_library(config['library']);original_ids={k:b.sha256 for k,b in blocks.items()}
        study=constructor();model,tokenizer=study.model,study.tokenizer
        model.eval();model.gradient_checkpointing_disable();base=frozen_hash(model)
        def load(path,expected):
            guard();saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['base_hash']!=base or saved['manifest_sha256']!=expected:raise ValueError('Checkpoint identity')
            restore_adapter(model,saved['adapter'])
        def generate(messages,cap,label):
            guard();t=time.monotonic()
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            ids=tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
            if ids.shape[1]+cap>8192:raise ValueError('Context overflow')
            with torch.inference_mode():
                output=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,use_cache=True,
                                      max_new_tokens=cap,pad_token_id=tokenizer.eos_token_id)
            ts=output[0,ids.shape[1]:].tolist();answer=tokenizer.decode(ts,skip_special_tokens=True)
            with (out/'local_replies.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(dict(label=label,prompt=messages,answer=answer,tokens=len(ts),hit_cap=len(ts)==cap,seconds=time.monotonic()-t))+'\n')
            return answer
        def execute(task,bank,mapping,label,reference=False):
            e=episode(task,bank,specs,mapping)
            while not e.finished:
                guard()
                action=teacher(e.observation()) if reference else generate([dict(role='user',content=prompt(e.observation()))],64,label)
                e.execute(action)
            result=e.result();trace=result.pop('trace');decisions=result.pop('decision_trace')
            path=label+'.trace.json.gz'
            (out/path).write_bytes(gzip.compress(json.dumps(dict(trace=trace,decisions=decisions),separators=(',',':')).encode(),mtime=0))
            result['trace_path']=path;return result
        calibration=tasks(11000000,[3,3,3,3,5,5,5,5])
        reserved=tasks(12000000,[3]*8+[5]*8+[8]*8+[12]*8)
        retention=tasks(13000000,[3]*8+[5]*8)
        save(out/'partitions.json',dict(calibration=calibration,reserved=reserved,retention=retention))
        mapping=config['mapping'];identity=[[1.,0.],[0.,1.]]
        current=blocks['move'];history=[];budgets=dict(turns=12,probe_commands=32,author_calls=4,fit_updates=4096,evaluations=4)
        seed='113'
        def acquisition(label):
            load(config['executives'][seed],config['executive_manifest'])
            bank=dict(blocks,move=current);measured=[]
            for i,t in enumerate(calibration):
                report(status='calibration',label=label,completed=i,total=len(calibration))
                r=execute(t,bank,mapping,f'{label}_{i}');measured.append(r)
            save(out/(label+'.json'),measured)
            # Bounded evidence: retain full traces privately, expose early actions/events.
            visible=[]
            for task,row in zip(calibration,measured):
                trace=json.loads(gzip.decompress((out/row['trace_path']).read_bytes()))
                visible.append(dict(task=task,success=row['task_success'],error=row['error'],violations=row['violations'],
                                    decisions=trace['decisions'][:4],motor_trace=trace['trace'][:4]))
            return dict(success=sum(r['task_success'] for r in measured),total=len(measured),episodes=visible)
        report(status='initial_calibration')
        history.append(dict(initial_acquisition=acquisition('initial')))
        for turn in range(1,13):
            guard();budgets['turns']=13-turn;report(status='astra',turn=turn)
            text=packet(blocks['move'],specs,history,budgets)
            def heartbeat():report(status='astra',turn=turn);return stopped()
            answer,audit=collect(config['cli'],config['catalog'],config['instructions'],out/f'turn_{turn}',text,1200,heartbeat)
            entry=dict(assistant=answer,usage=audit['usage']);finish=False
            try:
                a=parse_action(answer);tool=a['tool'];result={}
                if tool=='probe':
                    if len(a['commands'])>budgets['probe_commands']:raise ValueError('Probe budget exhausted')
                    result=dict(observations=probe(a['commands'],mapping));budgets['probe_commands']-=len(a['commands'])
                elif tool in ('install_wspl','install_native'):
                    proposed=candidate(a,blocks['move'],specs['move']);current=proposed
                    result=dict(installed=True,weights=current.to_dict())
                elif tool=='author':
                    if budgets['author_calls']<=0:raise ValueError('Author budget exhausted')
                    budgets['author_calls']-=1;load(config['author_adapter'],config['author_manifest'])
                    source=generate([dict(role='user',content='Write one WSPL Core module named Block. Inputs x,y Real[-2,2] normalized_x, normalized_y; outputs vx,vy,grip Real[-8,8] with matching normalized units. Use affine arithmetic, clip, relu, min, max. At most27compiled parameters. Return only source. Request: '+a['request'])],512,f'author_{turn}')
                    result['source']=source
                    try:
                        proposed=candidate(dict(tool='install_wspl',source=source),blocks['move'],specs['move']);current=proposed;result.update(installed=True,weights=current.to_dict())
                    except (ValueError,KeyError,TypeError) as err:result.update(installed=False,error=str(err))
                elif tool=='fit':
                    if a['updates']>budgets['fit_updates']:raise ValueError('Fit budget exhausted')
                    budgets['fit_updates']-=a['updates'];report(status='fitting',turn=turn)
                    proposed,fit=fit_replacement(current,a['rows'],700+turn,a['updates'],guard)
                    # Fitted artifacts must pass the same admission conditions.
                    candidate(dict(tool='install_native',layers=proposed.to_dict()['layers']),proposed,specs['move'])
                    current=proposed;result=dict(installed=True,fit=fit,weights=current.to_dict())
                elif tool=='evaluate':
                    if budgets['evaluations']<=0:raise ValueError('Evaluation budget exhausted')
                    budgets['evaluations']-=1;result=acquisition(f'calibration_{turn}')
                elif tool=='finish':finish=True;result=dict(frozen=True)
                entry['feedback']=dict(accepted=True,result=result)
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as err:
                entry['feedback']=dict(accepted=False,error=str(err))
            entry['current_sha256']=current.sha256;history.append(entry)
            save(out/'movement_last.weights.json',current.to_dict());save(out/'history.json',history);save(out/'budgets.json',budgets)
            if finish:break
        current.save(out/'movement_final.weights.json')
        # No further Astra calls or local fitting after artifact freeze.
        save(out/'frozen.json',dict(movement=current.sha256,executives={s:file_hash(p) for s,p in config['executives'].items()},original_library=original_ids))
        outcomes=[]
        for seed,path in config['executives'].items():
            load(path,config['executive_manifest'])
            for arm,block in [('unchanged',blocks['move']),('repaired',current)]:
                for panel,rows,mechanism in [('changed',reserved,mapping),('original',retention,identity)]:
                    for i,t in enumerate(rows):
                        report(status='final_evaluation',seed=seed,arm=arm,panel=panel,completed=i,total=len(rows))
                        row=execute(t,dict(blocks,move=block),mechanism,f'{arm}_{seed}_{panel}_{i}')
                        outcomes.append(dict(seed=seed,arm=arm,panel=panel,index=i,**row))
                        with (out/'scores.jsonl').open('a') as f:f.write(json.dumps(outcomes[-1])+'\n')
        # Supplied analytic inverse is evaluation-only and never enters agent context.
        ref=construct(config['reference_source'],specs['move'])
        for panel,rows,mechanism,block in [('changed',reserved,mapping,ref),('original',retention,identity,blocks['move'])]:
            for i,t in enumerate(rows):
                row=execute(t,dict(blocks,move=block),mechanism,f'reference_{panel}_{i}',True)
                outcomes.append(dict(seed='reference',arm='reference',panel=panel,index=i,**row))
        summary={}
        for row in outcomes:
            k='/'.join((row['seed'],row['arm'],row['panel']));s=summary.setdefault(k,dict(n=0,success=0))
            s['n']+=1;s['success']+=row['task_success']
        paired={}
        for seed in config['executives']:
            for panel in ('changed','original'):
                old={r['index']:r['task_success'] for r in outcomes if r['seed']==seed and r['arm']=='unchanged' and r['panel']==panel}
                new={r['index']:r['task_success'] for r in outcomes if r['seed']==seed and r['arm']=='repaired' and r['panel']==panel}
                assert old.keys()==new.keys()
                paired[seed+'/'+panel]=dict(n=len(old),improvements=sum(new[i] and not old[i] for i in old),
                     regressions=sum(old[i] and not new[i] for i in old),gain=(sum(new.values())-sum(old.values()))/len(old))
        reference_passed=all(r['task_success'] for r in outcomes if r['arm']=='reference')
        frozen=json.loads((out/'frozen.json').read_text())
        assert all(file_hash(config['executives'][s])==h for s,h in frozen['executives'].items())
        assert frozen_hash(model)==base and original_ids=={k:b.sha256 for k,b in blocks.items()}
        save(out/'scores.json',outcomes)
        result=dict(status='complete',summary=summary,author_turns=len(history)-1,remaining=budgets,
                    paired=paired,reference_passed=reference_passed,
                    feasibility_passed=reference_passed and all(paired[s+'/changed']['gain']>=.2 for s in config['executives']),
                    frozen_base_and_executives=True,original_library_unchanged=True,
                    deployment_has_frontier_agent=False,elapsed_seconds=time.monotonic()-started)
        save(out/'results.json',result);report(status='complete',results=result)
    except BaseException as err:
        report(status='paused' if isinstance(err,InterruptedError) else 'failed',error=repr(err));raise
