"""Unchanged versus AdamW continuation versus native Astra adapter repair."""
import json,time,random,subprocess,zipfile
from pathlib import Path
import numpy as np
import torch
from skillforge.grid_repair import weights,load_weights,validate,evaluate,diagnostics,EDITABLE
from skillforge.grid_movement import encode_task
from skillforge.grid_copy import batch_tensors,copy_loss,checkpoint
from skillforge.relu_adapter import ResidualReLU,attached
from skillforge.arc_eval import model_hash
from skillforge.arc_authoring import read_bank,parse_reply
from skillforge.codex_one_shot import collect
from skillforge.joint_attention import prompt
from skillforge.diffusion_edit import linux_path
from skillforge.go_replay import sha
from skillforge.reliable_json import save,status
from run_attention_probe import construct


def run(constructor,output_dir,config):
    out=Path(output_dir).resolve();receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('receipt')
    def resources():
        for p,h in config['resources'].items():
            if sha(p)!=h:raise ValueError('resource changed '+p)
    resources();subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['runtime_verify']),'--verify',linux_path(config['linux_identity'])],check=True,capture_output=True,timeout=90)
    start=time.monotonic();state=dict(status='loading',manifest_sha256=receipt['manifest_sha256'])
    def snapshot():state['active_seconds']=time.monotonic()-start;status(out/'status.json',state)
    def stop():
        p=out/'control.json'
        return time.monotonic()-start>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def check():
        if stop():raise RuntimeError('stop/guard')
    try:
        study=constructor();base_hash=model_hash(study.model)
        parent=torch.load(config['parent_adapter'],map_location='cpu',weights_only=False)
        adapter=ResidualReLU().to('cuda');adapter.load_state_dict(parent['model']);initial=weights(adapter)
        acquired=json.loads(Path(config['acquisition']).read_text());demos=json.loads(Path(config['demonstrations']).read_text())
        # Independent gradient continuation: same available examples, saved optimizer history.
        opt=torch.optim.AdamW(adapter.parameters(),lr=.001,weight_decay=0,foreach=False);opt.load_state_dict(parent['optimizer'])
        for group in opt.param_groups:group['lr']=.001
        rng=random.Random(431);move=[encode_task(study.tokenizer,r,r['task'],demos) for r in acquired if r['task']=='movement'];copy=[encode_task(study.tokenizer,r,r['task'],demos) for r in acquired if r['task']=='copy']
        state.update(status='gradient_control',step=0);snapshot();last_save=time.monotonic();step=0
        with attached(study.model,adapter), (out/'gradient_training.jsonl').open('w') as log:
            for step in range(1,1025):
                if stop():checkpoint(out/'gradient_last.pt',adapter,opt,step-1,receipt['manifest_sha256'],[],rng);check()
                mi=[rng.randrange(len(move)) for _ in range(3)];ci=rng.randrange(len(copy))
                batch=batch_tensors([move[i] for i in mi]+[copy[ci]],study.tokenizer.eos_token_id,'cuda')
                opt.zero_grad(set_to_none=True);loss=copy_loss(study.model,batch)
                if not torch.isfinite(loss):raise ValueError('nonfinite gradient control')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True);opt.step()
                entry=dict(step=step,nll=float(loss.detach()),grad_norm=float(norm),movement_indices=mi,copy_index=ci)
                log.write(json.dumps(entry)+'\n');log.flush();state.update(step=step,nll=entry['nll']);snapshot()
                if time.monotonic()-last_save>=1800:checkpoint(out/'gradient_last.pt',adapter,opt,step,receipt['manifest_sha256'],[],rng);last_save=time.monotonic()
        opt.zero_grad(set_to_none=True);checkpoint(out/'gradient_last.pt',adapter,opt,step,receipt['manifest_sha256'],[],rng)
        gradient=weights(adapter);np.savez_compressed(out/'gradient.npz',**gradient);del opt
        # Author starts from the original adapter, not the gradient continuation.
        load_weights(adapter,initial);adapter.eval().requires_grad_(False)
        state['status']='acquisition_diagnostics';snapshot()
        initial_eval=evaluate(study,adapter,acquired,demos,check);diag=diagnostics(study,adapter,acquired,demos,check)
        save(out/'acquisition_initial.json',initial_eval)
        folder=out/'author';folder.mkdir();current=folder/'current.npz';bank={**initial,**diag};np.savez_compressed(current,**bank)
        guide=Path(config['guide']).read_text();packet=dict(examples=acquired,initial=initial_eval,tensors={k:list(v.shape) for k,v in bank.items()})
        history=[];turns=[];threads=set();cli=json.loads(Path(config['local_paths']).read_text())['codex_cli']
        for turn_index in range(6):
            check();turn=folder/f'turn_{turn_index:02d}';state.update(status='authoring',turn=turn_index);snapshot();call_start=time.monotonic()
            def heartbeat():state['call_active_seconds']=time.monotonic()-call_start;snapshot();return stop()
            answer,audit=collect(cli,config['catalog'],config['instructions'],turn,prompt(guide,packet,history,turn_index),1200,heartbeat)
            if audit['thread_id'] in threads:raise ValueError('reused author thread')
            threads.add(audit['thread_id']);entry=dict(step=turn_index,collection=audit);turns.append(entry)
            proposed=None;feedback=dict(accepted=False)
            try:
                action,source=parse_reply(answer,'native')
                if action=='final':entry['status']='final';break
                if action!='construct':raise ValueError('inspect via Python prints')
                candidate,feedback=construct(source,current,turn,config);feedback['accepted']=False
                if feedback['exit_code']==0:
                    proposed=read_bank(candidate,bank);validate(proposed,{**bank,**initial})
            except (ValueError,OSError,EOFError,zipfile.BadZipFile) as e:feedback=dict(accepted=False,error=str(e));proposed=None
            if proposed is not None:
                load_weights(adapter,proposed);state['status']='acquisition_feedback';snapshot()
                measured=evaluate(study,adapter,acquired,demos,check)
                diag=diagnostics(study,adapter,acquired,demos,check)
                bank={**{k:proposed[k] for k in EDITABLE},**diag};np.savez_compressed(current,**bank)
                feedback.update(accepted=True,evaluation=measured)
            entry['status']='accepted' if feedback.get('accepted') else 'rejected';save(turn/'feedback.json',feedback)
            history.append(dict(assistant=answer,feedback=feedback));state['turns']=turns;snapshot()
        authored={k:bank[k] for k in EDITABLE};np.savez_compressed(folder/'final.npz',**authored)
        save(folder/'session.json',dict(turns=turns,selection='last accepted; no best-of selection'))
        np.savez_compressed(out/'unchanged.npz',**initial)
        frozen={name:sha(path) for name,path in [('unchanged',out/'unchanged.npz'),('gradient',out/'gradient.npz'),('astra',folder/'final.npz')]};save(out/'frozen.json',frozen)
        # First reserved predictions occur after both repairs are frozen.
        reserved=json.loads(Path(config['reserved']).read_text());summary={};records={}
        for name,tensors in [('unchanged',initial),('gradient',gradient),('astra',authored)]:
            load_weights(adapter,tensors);summary[name]={};records[name]={}
            for split,rows in reserved.items():
                state.update(status='evaluating',condition=name,split=split);snapshot();measured=evaluate(study,adapter,rows,demos,check)
                records[name][split]=measured['records'];summary[name][split]=dict(n=len(rows),exact=sum(r['exact'] for r in measured['records']))
                save(out/f'{name}_{split}.json',measured)
        paired={}
        for name in ('gradient','astra'):
            paired[name]={}
            for split in reserved:
                a=records['unchanged'][split];b=records[name][split]
                paired[name][split]=dict(corrected=sum(not x['exact'] and y['exact'] for x,y in zip(a,b)),broken=sum(x['exact'] and not y['exact'] for x,y in zip(a,b)))
        if model_hash(study.model)!=base_hash or any(p.grad is not None for p in study.model.parameters()):raise ValueError('foundation mutation')
        for name,path in [('unchanged',out/'unchanged.npz'),('gradient',out/'gradient.npz'),('astra',folder/'final.npz')]:
            if sha(path)!=frozen[name]:raise ValueError('frozen artifact mutation')
        repair_signal=(summary['astra']['fresh_L']['exact']-summary['unchanged']['fresh_L']['exact']>=16 and
                       summary['astra']['larger']['exact']-summary['unchanged']['larger']['exact']>=8 and
                       paired['astra']['copy']['broken']<=3 and summary['astra']['copy']['exact']>=61)
        resources();state.update(status='complete',summary=summary,paired=paired,frozen=frozen,foundation_sha256=base_hash,descriptive_repair_signal=repair_signal,
            next_step='No automatic follow-on. Report all arms and construction failures.')
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
