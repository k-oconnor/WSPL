"""One bounded Astra session to repair a fixed learned spatial adapter."""
import json,time,subprocess
from pathlib import Path
from contextlib import nullcontext
import torch
from skillforge.capability_bridge import modules,attached,validate_plan
from skillforge.capability_codec import evaluate
from skillforge.arc_eval import model_hash
from skillforge.arc_authoring import parse_reply
from skillforge.codex_one_shot import collect
from skillforge.diffusion_edit import linux_path
from skillforge.go_replay import sha
from skillforge.reliable_json import save,status
from skillforge.integration_repair import scaled,compact,choose_strength,assess
from run_prose_degradation import next_token_loss
from run_capability6 import construct


def author_prompt(guide,packet,history,turn):
    return guide+'\nACQUISITION PACKET:\n'+json.dumps(packet)+'\nVISIBLE TRANSCRIPT:\n'+json.dumps(history)+f'\nResponse {turn+1}/4. Return one fenced Python program or FINAL.'


def measure(study,plan,data,check):
    bank=None if plan is None else modules(plan).eval().requires_grad_(False)
    bank_hash=None if bank is None else model_hash(bank)
    with nullcontext() if bank is None else attached(study.model,bank):
        grids={name:evaluate(study,rows,check) for name,rows in data['grids'].items()}
        prose=[]
        for row in data['prose']:
            check();ids=torch.tensor([study.tokenizer.encode(row['text'],add_special_tokens=False)[:513]],device='cuda')
            if ids.shape[1]<2:raise ValueError('short text')
            loss,n=next_token_loss(study.model,ids);prose.append(dict(id=row['id'],loss_sum=loss,targets=n))
    if bank is not None and model_hash(bank)!=bank_hash:raise ValueError('adapter changed')
    n=sum(r['targets'] for r in prose)
    return dict(grids=grids,prose=prose,prose_targets=n,prose_nll=sum(r['loss_sum'] for r in prose)/n)


def run(constructor,output_dir,config):
    out=Path(output_dir).resolve();start=time.monotonic();state=dict(status='verifying')
    receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('source receipt')
    state['manifest_sha256']=receipt['manifest_sha256']
    def snapshot():state['active_seconds']=time.monotonic()-start;status(out/'status.json',state)
    def stopped():
        p=out/'control.json'
        return time.monotonic()-start>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def check():
        snapshot()
        if stopped():raise RuntimeError('stop/guard')
    def resources():
        for p,h in config['resources'].items():
            if sha(p)!=h:raise ValueError('resource changed '+p)
    try:
        snapshot();resources()
        subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['runtime_verify']),'--verify',linux_path(config['linux_identity'])],check=True,capture_output=True,timeout=90)
        state['status']='loading';snapshot();study=constructor();before=model_hash(study.model)
        original=json.loads(Path(config['adapter']).read_text());acq=json.loads(Path(config['acquisition']).read_text())
        state['status']='acquisition_controls';snapshot()
        foundation=measure(study,None,acq,check);save(out/'foundation_acquisition.json',foundation)
        controls={}
        for alpha in (1.,.75,.5,.25):
            key=str(alpha);state['condition']='strength_'+key;snapshot()
            controls[key]=measure(study,original if alpha==1 else scaled(original,alpha),acq,check)
            save(out/('strength_'+key+'_acquisition.json'),controls[key])
        chosen=choose_strength(controls);scalar=original if chosen=='1.0' else scaled(original,float(chosen))
        packet=dict(objective='Preserve movement while recovering ordinary prose behavior; adapter always enabled.',
            examples={k:v[:4] for k,v in acq['grids'].items()},prose_examples=[r['text'][:1200] for r in acq['prose'][:2]],
            foundation=compact(foundation),strength_controls={k:compact(v) for k,v in controls.items()},chosen_strength=chosen,
            parameters=validate_plan(original)['parameters'],training='none; recipe changes are rejected')
        guide=Path(config['guide']).read_text();cli=json.loads(Path(config['local_paths']).read_text())['codex_cli']
        current=original;history=[];turns=[];threads=set()
        for i in range(4):
            check();state.update(status='authoring',turn=i+1);snapshot();folder=out/f'turn_{i:02d}'
            def heartbeat():snapshot();return stopped()
            answer,audit=collect(cli,config['catalog'],config['instructions'],folder,author_prompt(guide,packet,history,i),1200,heartbeat)
            if audit['thread_id'] in threads:raise ValueError('reused thread')
            threads.add(audit['thread_id']);entry=dict(turn=i+1,collection=audit);turns.append(entry)
            try:
                action,source=parse_reply(answer,'native')
                if action=='final':entry['status']='final';break
                if action!='construct':raise ValueError('Python or FINAL required')
                proposed,result=construct(source,current,packet,'wspl',folder,config)
                if proposed is not None and proposed['recipe']!=original['recipe']:raise ValueError('no training; preserve recipe')
            except ValueError as e:proposed=None;result=dict(error=str(e))
            if proposed is None:entry['status']='rejected'
            else:
                current=proposed;state['status']='acquisition_feedback';snapshot()
                measured=measure(study,current,acq,check);save(folder/'acquisition.json',measured)
                result.update(accepted=True,metrics=compact(measured));entry['status']='accepted'
            save(folder/'feedback.json',result);history.append(dict(assistant=answer,feedback=result))
            if model_hash(study.model)!=before:raise ValueError('base changed')
        save(out/'final.json',current);save(out/'scalar.json',scalar)
        save(out/'session.json',dict(turns=turns,selected_strength=chosen,selection='last accepted artifact'))
        frozen=dict(repair=sha(out/'final.json'),scalar=sha(out/'scalar.json'));save(out/'frozen.json',frozen)
        # Reserved rows opened only after final artifact identity has been frozen.
        reserved=json.loads(Path(config['reserved']).read_text());reports={}
        for arm,plan in [('foundation',None),('original',original),('scalar',scalar),('repair',current)]:
            state.update(status='reserved_evaluation',condition=arm);snapshot()
            reports[arm]=measure(study,plan,reserved,check);save(out/(arm+'_reserved.json'),reports[arm])
        if model_hash(study.model)!=before or any(p.grad is not None for p in study.model.parameters()):raise ValueError('base changed')
        for name,h in frozen.items():
            if sha(out/('final.json' if name=='repair' else 'scalar.json'))!=h:raise ValueError('frozen artifact changed')
        resources();state.update(status='complete',summaries={k:compact(v) for k,v in reports.items()},decision=assess(reports),selected_strength=chosen,frozen=frozen)
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
