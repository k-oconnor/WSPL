"""One narrow native WSP+ASP feasibility session; no optimizer training."""
import json,time,subprocess,zipfile
from pathlib import Path
import numpy as np
from skillforge.arc_eval import model_hash
from skillforge.arc_attention import score_boundary
from skillforge.arc_authoring import read_bank,parse_reply
from skillforge.joint_attention import make_reference,validate,weight_edit,metrics,prompt
from skillforge.codex_one_shot import collect
from skillforge.diffusion_edit import linux_path
from skillforge.go_replay import sha
from skillforge.reliable_json import save,status
from run_attention_probe import construct


def session(reference,folder,packet,guide,collector,compiler,evaluate,checkpoint):
    folder.mkdir();bank={k:v.copy() for k,v in reference.items()};current=folder/'current.npz'
    np.savez_compressed(current,**bank);history=[];turns=[]
    for step in range(6):
        turn=folder/f'turn_{step:02d}';answer,audit=collector(turn,prompt(guide,packet,history,step))
        entry=dict(step=step,collection=audit);turns.append(entry)
        try:
            action,source=parse_reply(answer,'native')
            if action=='final':entry['status']='final';break
            if action!='construct':raise ValueError('use Python prints for inspections')
            candidate,feedback=compiler(source,current,turn);feedback['accepted']=False
            if feedback['exit_code']==0:
                proposed=read_bank(candidate,reference);validate(proposed,reference)
                # Model/infrastructure failures must abort rather than look like a rejected proposal.
            else:proposed=None
        except (ValueError,OSError,EOFError,zipfile.BadZipFile) as e:
            feedback=dict(accepted=False,error=str(e));proposed=None
        if proposed is not None:
            measured=evaluate(proposed);bank=proposed;np.savez_compressed(current,**bank)
            feedback.update(accepted=True,evaluation=measured)
        entry['status']='accepted' if feedback['accepted'] else 'rejected'
        save(turn/'feedback.json',feedback);history.append(dict(assistant=answer,feedback=feedback));checkpoint(step,turns)
    np.savez_compressed(folder/'final.npz',**bank)
    save(folder/'session.json',dict(turns=turns,selection='last accepted, not best',sha256=sha(folder/'final.npz')))
    return bank


def run(constructor,output_dir,config):
    out=Path(output_dir).resolve();receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('receipt')
    def resources():
        for p,h in config['resources'].items():
            if sha(p)!=h:raise ValueError('changed resource '+p)
    resources()
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['runtime_verify']),'--verify',linux_path(config['linux_identity'])],check=True,capture_output=True,timeout=90)
    started=time.monotonic();state=dict(status='loading',manifest_sha256=receipt['manifest_sha256'])
    def snapshot():state['active_seconds']=time.monotonic()-started;status(out/'status.json',state)
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def check():
        if stop():raise RuntimeError('stop/guard')
    try:
        study=constructor();before=model_hash(study.model)
        if before!=config['model_sha256']:raise ValueError('model identity')
        with np.load(config['parent_bank'],allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
        acquired=json.loads(Path(config['acquisition']).read_text())
        reference,base,initial=make_reference(study,parent,acquired)
        np.savez_compressed(out/'reference.npz',**reference)
        save(out/'acquisition_baseline.json',base);save(out/'acquisition_initial_asp.json',initial)
        packet=dict(examples=acquired,baseline=base,initial_asp=initial,tensors={k:list(v.shape) for k,v in reference.items()})
        guide=Path(config['guide']).read_text(encoding='utf-8');cli=json.loads(Path(config['local_paths']).read_text())['codex_cli'];threads=set()
        def collector(folder,text):
            check();state.update(status='authoring',turn=folder.name);snapshot();t=time.monotonic()
            def heartbeat():state['call_active_seconds']=time.monotonic()-t;snapshot();return stop()
            answer,audit=collect(cli,config['catalog'],config['instructions'],folder,text,1200,heartbeat)
            if audit['thread_id'] in threads:raise ValueError('reused thread')
            threads.add(audit['thread_id']);return answer,audit
        def evaluate(bank):
            records=[]
            with weight_edit(study.model,bank):
                for row in acquired:check();records.append(score_boundary(study,row,bank,'retrieval'))
            if model_hash(study.model)!=before:raise ValueError('base restoration')
            return dict(records=records,metrics=metrics(records,base,acquired))
        def checkpoint(step,turns):state.update(step=step,turns=turns);snapshot()
        candidate=session(reference,out/'author',packet,guide,collector,lambda s,p,t:construct(s,p,t,config),evaluate,checkpoint)
        frozen=sha(out/'author/final.npz');save(out/'frozen.json',dict(sha256=frozen,selection='last accepted'))
        # Reserved examples are first opened only after freezing the final artifact.
        rows=json.loads(Path(config['reserved']).read_text());outputs={};summary={}
        for name,bank,weights,mode in [('untouched',reference,False,'off'),('initial_asp',reference,False,'retrieval'),
                                     ('joint',candidate,True,'retrieval'),('final_wsp_only',candidate,True,'off'),('final_asp_only',candidate,False,'retrieval')]:
            state.update(status='evaluating',variant=name);snapshot();records=[]
            with weight_edit(study.model,bank,weights):
                for row in rows:check();records.append(score_boundary(study,row,bank,mode))
            outputs[name]=records;summary[name]=metrics(records,outputs['untouched'],rows);save(out/(name+'_reserved.json'),records)
        if sha(out/'author/final.npz')!=frozen or model_hash(study.model)!=before:raise ValueError('frozen identity')
        resources();state.update(status='complete',summary=summary,final_sha256=frozen,model_sha256=before)
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
