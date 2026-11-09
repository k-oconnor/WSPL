"""Matched native/WSPL/LoRA sessions, followed by frozen test evaluation."""
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np
from skillforge.arc_authoring import ORDER,MAX_STEPS,read_bank,parse_reply,inspect_bank,statistics,prompt
from skillforge.smollm_adapter import TARGETS,initial_bank,effective_weights
from skillforge.arc_eval import score_row,model_hash
from skillforge.go_replay import sha
from skillforge.codex_one_shot import collect
from skillforge.diffusion_edit import linux_path
from run_diffusion_replay import save


def construct(arm,source,parent,base,folder,config):
    path=folder/('proposal.py' if arm=='native' else 'proposal.wspl');path.write_text(source,encoding='utf-8')
    receipt=folder/'construction.json';output=folder/'constructed'
    if arm=='native':
        command=['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['python_worker']),
            '--source',linux_path(path),'--parent',linux_path(parent),'--output',linux_path(output),
            '--runtime',config['python_runtime'],'--receipt',linux_path(receipt)]
    else:
        command=[sys.executable,config['wspl_worker'],'--source',str(path),'--parent',str(parent),'--base',str(base),
            '--mode','lora' if arm=='lora' else 'direct','--output',str(output),'--receipt',str(receipt)]
    try:run=subprocess.run(command,capture_output=True,text=True,timeout=180)
    except subprocess.TimeoutExpired:return None,dict(exit_code=1,stdout='',stderr='construction timeout; state unchanged')
    if not receipt.exists():raise RuntimeError('construction infrastructure failure: '+run.stderr[-1000:])
    feedback=json.loads(receipt.read_text())
    # Archive full worker output, but do not expose a native-only tensor-print channel.
    feedback['stdout']=''
    feedback['stderr']=feedback.get('stderr','')[-1000:]
    return output/'candidate.npz',feedback


def author_session(name,arm,out,base,packet,guide,config,collector,compiler,evaluate,stop,snapshot,threads):
    folder=out/name;folder.mkdir();mode='lora' if arm=='lora' else 'direct';expected=initial_bank(base,mode)
    bank={k:v.copy() for k,v in expected.items()};current=folder/'current.npz';np.savez_compressed(current,**bank)
    history=[];row=dict(name=name,arm=arm,status='authoring',turns=[]);snapshot(row)
    initial=dict(packet,current_tensors=statistics(bank))
    for step in range(MAX_STEPS):
        if stop():raise RuntimeError('authoring stopped')
        turn=folder/f'turn_{step:02d}';text=prompt(guide,initial,arm,history,step)
        answer,audit=collector(turn,text,stop)
        if audit['thread_id'] in threads:raise ValueError('thread reused')
        threads.add(audit['thread_id']);entry=dict(step=step,collection=audit);row['turns'].append(entry)
        try:action,payload=parse_reply(answer,arm)
        except ValueError as e:action='invalid';feedback=dict(accepted=False,error=str(e))
        if action=='final':entry['status']='final';row['termination']='author_final';break
        if action=='inspect':
            try:feedback=dict(accepted=False,inspection=inspect_bank(bank,payload))
            except ValueError as e:feedback=dict(accepted=False,error=str(e))
        elif action=='construct':
            candidate,feedback=compiler(arm,payload,current,out/'base.npz',turn,config);feedback['accepted']=False
            if feedback['exit_code']==0 and candidate is not None:
                try:
                    proposed=read_bank(candidate,expected);effective,metrics=effective_weights(base,proposed,mode)
                except (ValueError,OSError,EOFError,zipfile.BadZipFile) as e:feedback['stderr']=str(e)
                else:
                    # Candidate accepted only after interface checks; failures in evaluation stop the study.
                    bank=proposed;np.savez_compressed(current,**bank);feedback['accepted']=True;feedback['effective_edit']=metrics
                    feedback['validation']=evaluate(effective,turn/'validation.jsonl')
        entry.update(status='accepted' if feedback.get('accepted') else 'inspection' if action=='inspect' else 'rejected',feedback=feedback)
        save(turn/'feedback.json',feedback);history.append(dict(assistant=answer,feedback=feedback));snapshot(row)
    else:row['termination']='response_budget_exhausted'
    final=folder/'final.npz';np.savez_compressed(final,**bank)
    row.update(status='frozen',sha256=sha(final),path=final.relative_to(out).as_posix(),mode=mode);snapshot(row)
    return row


def run(constructor,output_dir,config):
    import torch
    out=Path(output_dir).resolve();receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('approved receipt required')
    def verify_resources():
        for path,h in config['resources'].items():
            if sha(path)!=h:raise ValueError('resource changed: '+path)
    verify_resources()
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['runtime_verify']),'--verify',linux_path(config['linux_identity'])],check=True,capture_output=True,timeout=90)
    started=time.monotonic();state=dict(status='loading',trials=[],manifest_sha256=receipt['manifest_sha256']);save(out/'status.json',state)
    def snapshot():state['active_seconds']=time.monotonic()-started;save(out/'status.json',state)
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    try:
        study=constructor();original_hash=model_hash(study.model);params=dict(study.model.named_parameters())
        base={k:params[name].detach().cpu().numpy().copy() for k,name in TARGETS.items()};np.savez_compressed(out/'base.npz',**base)
        baseline=json.loads(Path(config['baseline_results']).read_text())
        if original_hash!=baseline['unchanged_model_sha256']:raise ValueError('baseline weights differ')
        dev=[json.loads(x) for x in Path(config['validation']).read_text(encoding='utf-8').splitlines()]
        examples=json.loads(Path(config['examples']).read_text(encoding='utf-8'))
        def install(weights):
            with torch.no_grad():
                for k,name in TARGETS.items():params[name].copy_(torch.from_numpy(weights[k]).to('cuda'))
        def evaluate(weights,path,rows=dev,feedback=True):
            install(weights);records=[]
            with path.open('w',encoding='utf-8') as log:
                for row in rows:
                    if stop():raise RuntimeError('evaluation stopped')
                    record=score_row(study.model,study.tokenizer,row);records.append(record);log.write(json.dumps(record)+'\n')
                    if len(records)%16==0:
                        state.update(evaluation_examples_completed=len(records),evaluation_examples_total=len(rows));snapshot()
            summary=dict(n=len(records),accuracy=sum(r['correct'] for r in records)/len(records),raw_accuracy=sum(r['raw_correct'] for r in records)/len(records))
            if feedback:summary['fixed_examples']=[dict(question=r,score=s) for r,s in zip(rows[:16],records[:16])]
            return summary
        native_baseline=evaluate(base,out/'baseline_validation.jsonl')
        previous=[json.loads(x) for x in Path(config['baseline_predictions']).read_text().splitlines()]
        fresh=[json.loads(x) for x in (out/'baseline_validation.jsonl').read_text().splitlines()]
        if previous!=fresh:raise ValueError('baseline validation replay mismatch')
        packet=dict(model='SmolLM2-135M-Instruct',architecture=config['architecture'],training_examples=examples,
            baseline_validation=native_baseline,targets=TARGETS,rank=8,scale=1,max_relative_frobenius=.1)
        guide=Path(config['guide']).read_text(encoding='utf-8');cli=json.loads(Path(config['local_paths']).read_text())['codex_cli'];threads=set();trials=[]
        for draw,arm in ORDER:
            install(base)
            if model_hash(study.model)!=original_hash:raise ValueError('non-target weights changed')
            name=f'{draw}_{arm}'
            def onrow(row):state.update(status='authoring',trials=trials+[row]);snapshot()
            def collector(folder,text,should_stop):
                start=time.monotonic();last=[0.]
                def heartbeat():
                    now=time.monotonic()
                    if now-last[0]>5:
                        last[0]=now;state.update(current_call=str(folder.relative_to(out)),call_active_seconds=now-start);snapshot()
                    return should_stop()
                try:return collect(cli,config['catalog'],config['instructions'],folder,text,1200,heartbeat)
                finally:state.pop('current_call',None);state.pop('call_active_seconds',None)
            row=author_session(name,arm,out,base,packet,guide,config,collector,construct,evaluate,stop,onrow,threads);trials.append(row)
        install(base)
        if model_hash(study.model)!=original_hash:raise ValueError('non-target weights changed')
        frozen={row['name']:{k:row[k] for k in ('path','sha256','mode')} for row in trials}
        save(out/'frozen_candidates.json',frozen)
        # No test rows are loaded or scored until all nine author sessions freeze.
        state.update(status='evaluating',trials=trials);snapshot();results={}
        for name in ['untouched']+list(frozen):
            if name=='untouched':weights=base
            else:
                row=frozen[name]
                if sha(out/row['path'])!=row['sha256']:raise ValueError('frozen candidate changed')
                bank=read_bank(out/row['path'],initial_bank(base,row['mode']));weights,_=effective_weights(base,bank,row['mode'])
            folder=out/name/'test';folder.mkdir(parents=True);results[name]={}
            for task,path in config['test_datasets'].items():
                state.update(evaluating=name,test_task=task);snapshot()
                rows=[json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]
                results[name][task]=evaluate(weights,folder/(task+'.jsonl'),rows,False)
            state['evaluations']=results;snapshot()
        install(base)
        if model_hash(study.model)!=original_hash:raise ValueError('base model identity changed')
        verify_resources();state.update(status='complete',evaluations=results,original_model_sha256=original_hash,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:
        state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
