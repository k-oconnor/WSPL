"""Bounded native versus native-plus-WSPL capability engineering screen."""
import json,time,subprocess
from pathlib import Path
import torch
from skillforge.capability_bridge import default_plan,validate_plan,modules,attached
from skillforge.capability_training import train_chunk,feedback,compact_feedback
from skillforge.capability_codec import encode,evaluate
from skillforge.arc_eval import model_hash
from skillforge.arc_authoring import parse_reply
from skillforge.codex_one_shot import collect
from skillforge.diffusion_edit import linux_path
from skillforge.go_replay import sha
from skillforge.reliable_json import save,status


def author_prompt(guide,interface,packet,history,turn):
    text=guide+'\nINTERFACE: '+interface+'\nACQUISITION PACKET:\n'+json.dumps(packet)+'\nVISIBLE TRANSCRIPT:\n'+json.dumps(history)
    text+=f'\nResponse {turn+1}/4. Return one fenced python program or FINAL.'
    if len(text.encode())>800000:raise ValueError('prompt budget')
    return text


def construct(source,plan,packet,interface,folder,config):
    (folder/'proposal.py').write_text(source,encoding='utf-8');save(folder/'request.json',dict(plan=plan,packet=packet,interface=interface))
    receipt=folder/'construction.json'
    args=['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['worker']),'--source',linux_path(folder/'proposal.py'),'--request',linux_path(folder/'request.json'),
          '--library',linux_path(Path('src').resolve()),'--output',linux_path(folder/'constructed'),'--runtime',config['python_runtime'],'--receipt',linux_path(receipt)]
    subprocess.run(args,capture_output=True,timeout=300)
    if not receipt.exists():raise RuntimeError('construction infrastructure failed')
    result=json.loads(receipt.read_text());candidate=None
    if result['exit_code']==0:
        p=folder/'constructed/candidate.json'
        if p.stat().st_size>256<<20:raise ValueError('candidate size limit')
        data=json.loads(p.read_text());validate_plan(data['plan']);candidate=data['plan'];result['events']=[{k:v for k,v in e.items() if k!='plan'} for e in data['events']]
        # Detailed immutable WSPL edit plans remain in candidate.json.
    return candidate,result


def run(constructor,output_dir,config):
    out=Path(output_dir).resolve();receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('receipt')
    def resources():
        for p,h in config['resources'].items():
            if sha(p)!=h:raise ValueError('resource changed '+p)
    resources();subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['runtime_verify']),'--verify',linux_path(config['linux_identity'])],check=True,capture_output=True,timeout=90)
    start=time.monotonic();state=dict(status='loading',manifest_sha256=receipt['manifest_sha256']);threads=set()
    def snapshot():state['active_seconds']=time.monotonic()-start;status(out/'status.json',state)
    def stopped():
        p=out/'control.json'
        return time.monotonic()-start>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def check():
        if stopped():raise RuntimeError('stop/guard')
    try:
        study=constructor();before=model_hash(study.model)
        train=json.loads(Path(config['train']).read_text());dev=json.loads(Path(config['dev']).read_text())
        encoded={k:[encode(study.tokenizer,r) for r in rows] for k,rows in train.items()}
        default=default_plan();validate_plan(default)
        packet=dict(training_counts={k:len(v) for k,v in train.items()},development_counts={k:len(v) for k,v in dev.items()},
            examples={k:v[:8] for k,v in train.items()},development_examples={k:v[:8] for k,v in dev.items()},default_parameters=validate_plan(default)['parameters'])
        guide=Path(config['guide']).read_text();cli=json.loads(Path(config['local_paths']).read_text())['codex_cli'];frozen={}
        for arm,interface in [('gradient','control'),('native_0','native'),('wspl_0','wspl'),('wspl_1','wspl'),('native_1','native')]:
            folder=out/arm;folder.mkdir();plan=default;budget=dict(updates=0,train_seconds=0.,targets=0);history=[];turns=[]
            for turn_index in range(4):
                check();state.update(arm=arm,turn=turn_index,status='design');snapshot()
                if interface!='control':
                    turn=folder/f'turn_{turn_index:02d}';call_start=time.monotonic()
                    def heartbeat():state['call_seconds']=time.monotonic()-call_start;snapshot();return stopped()
                    answer,audit=collect(cli,config['catalog'],config['instructions'],turn,author_prompt(guide,interface,dict(packet,budget=budget),history,turn_index),1200,heartbeat)
                    if audit['thread_id'] in threads:raise ValueError('reused thread')
                    threads.add(audit['thread_id']);entry=dict(turn=turn_index,collection=audit);turns.append(entry)
                    try:
                        action,source=parse_reply(answer,'native')
                        if action=='final':entry['status']='final';break
                        if action!='construct':raise ValueError('return Python or FINAL')
                        proposed,result=construct(source,plan,dict(packet,budget=budget),interface,turn,config)
                    except ValueError as e:proposed=None;result=dict(error=str(e))
                    if proposed is None:
                        entry['status']='rejected';save(turn/'feedback.json',result);history.append(dict(assistant=answer,feedback=result));continue
                    plan=proposed;entry['status']='accepted'
                else:turn=folder/f'chunk_{turn_index:02d}';turn.mkdir();result={}
                state['status']='training';snapshot()
                def progress(record):state.update(step=record['updates'],training_seconds=record['train_seconds'],loss=record['loss']);snapshot()
                plan=train_chunk(study,plan,train,encoded,budget,folder,receipt['manifest_sha256'],check,progress)
                state['status']='development';snapshot();measured=feedback(study,plan,dev,check);save(turn/'development.json',measured)
                result.update(accepted=True,development=compact_feedback(measured),budget=budget.copy(),parameters=validate_plan(plan)['parameters'])
                save(turn/'feedback.json',result)
                if interface!='control':history.append(dict(assistant=answer,feedback=result))
                if model_hash(study.model)!=before or any(p.grad is not None for p in study.model.parameters()):raise ValueError('base modified')
            save(folder/'final.json',plan);save(folder/'session.json',dict(turns=turns,budget=budget,selection='last accepted and trained'))
            frozen[arm]=sha(folder/'final.json');save(out/'frozen.json',frozen)
        # Final inputs are not opened until every arm is frozen.
        tests=json.loads(Path(config['test']).read_text());summaries={};records={}
        for arm in ['foundation']+list(frozen):
            from contextlib import nullcontext
            bank=None if arm=='foundation' else modules(json.loads((out/arm/'final.json').read_text()))
            if bank is not None:bank.eval().requires_grad_(False)
            summaries[arm]={};records[arm]={}
            with (nullcontext() if bank is None else attached(study.model,bank)):
                for split,rows in tests.items():
                    state.update(status='evaluating',arm=arm,split=split);snapshot();measured=evaluate(study,rows,check)
                    save(out/f'{arm}_{split}.json',measured);summaries[arm][split]=dict(n=measured['n'],exact=measured['exact']);records[arm][split]=measured['records']
            del bank
        decisions={}
        for arm in frozen:
            s=summaries[arm];b=summaries['foundation'];retention={}
            for split in ('copy4','copy6'):
                retention[split]=sum(x['exact'] and not y['exact'] for x,y in zip(records['foundation'][split],records[arm][split]))
            absolute=s['familiar']['exact']>=487 and s['novel']['exact']>=461 and s['larger']['exact']>=461 and s['composition']['exact']>=231
            gain=all(s[k]['exact']-b[k]['exact']>=103 for k in ('familiar','novel','larger'))
            kept=all(s[k]['exact']>=507 and retention[k]<=5 for k in ('copy4','copy6'))
            decisions[arm]=dict(capability_pass=absolute and gain and kept,absolute_pass=absolute,gain_pass=gain,copy_pass=kept,copy_regressions=retention)
            if sha(out/arm/'final.json')!=frozen[arm]:raise ValueError('artifact mutated')
        if model_hash(study.model)!=before:raise ValueError('foundation changed')
        resources();state.update(status='complete',summary=summaries,decisions=decisions,frozen=frozen,next_step='No automatic follow-on.')
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
