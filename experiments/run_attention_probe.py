"""Two-stage, one-question native attention authoring diagnostic."""
import json
from pathlib import Path
import subprocess
import time
import zipfile
import shutil
import numpy as np
from skillforge.attention_probe import validate, prefix_score, make_bank, author_prompt
from skillforge.arc_authoring import read_bank, parse_reply
from skillforge.arc_eval import model_hash
from skillforge.codex_one_shot import collect
from skillforge.diffusion_edit import linux_path
from skillforge.go_replay import sha
from skillforge.reliable_json import save, status as save_status


def construct(source, parent, turn, config):
    program=turn/'proposal.py';program.write_text(source,encoding='utf-8')
    output=turn/'constructed';receipt=turn/'construction.json'
    args=['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['python_worker']),
          '--source',linux_path(program),'--parent',linux_path(parent),'--output',linux_path(output),
          '--runtime',config['python_runtime'],'--receipt',linux_path(receipt)]
    result=subprocess.run(args,capture_output=True,text=True,timeout=180)
    if not receipt.exists(): raise RuntimeError('worker infrastructure failure: '+result.stderr[-1000:])
    feedback=json.loads(receipt.read_text());feedback['stdout']=feedback['stdout'][:12000];feedback['stderr']=feedback['stderr'][-2000:]
    return output/'candidate.npz',feedback


def session(stage, initial, reference, target, folder, guide, config, collector, evaluate, compiler, snapshot, frozen=None, resume=None):
    folder.mkdir();bank={k:v.copy() for k,v in initial.items()};current=folder/'current.npz'
    history=[];turns=[];start=0
    if resume is not None:
        if stage!='forced':raise ValueError('only interrupted forced stage supported')
        parent=Path(resume);parent_state=json.loads((parent/'status.json').read_text())
        if parent_state['status']!='incomplete' or parent_state['step']!=1:raise ValueError('unexpected recovery boundary')
        saved=read_bank(parent/'reference.npz',reference)
        if any(saved[k].tobytes()!=reference[k].tobytes() for k in reference):raise ValueError('reference drift')
        for step in range(2):
            src=parent/'forced'/f'turn_{step:02d}';dst=folder/src.name
            feedback=json.loads((src/'feedback.json').read_text());answer=(src/'answer.md').read_text(encoding='utf-8')
            if not feedback['accepted']:raise ValueError('unexpected rejected parent turn')
            bank=read_bank(src/'constructed/candidate.npz',reference);validate(bank,reference,stage)
            shutil.copytree(src,dst)
            history.append(dict(assistant=answer,feedback=feedback));turns.append(parent_state['turns'][step])
        saved_current=read_bank(parent/'forced/current.npz',reference)
        if any(saved_current[k].tobytes()!=bank[k].tobytes() for k in bank):raise ValueError('parent state mismatch')
        start=2
        save(folder/'recovery.json',dict(parent=str(parent),completed_turns_preserved=2,interrupted_call='forced/turn_02',interrupted_usage='unknown'))
    np.savez_compressed(current,**bank)
    for step in range(start,6):
        turn=folder/f'turn_{step:02d}'
        answer,audit=collector(turn,author_prompt(guide,target,stage,history,step))
        entry=dict(step=step,collection=audit);turns.append(entry)
        try:
            action,source=parse_reply(answer,'native')
            if action=='final':entry['status']='final';break
            if action!='construct':raise ValueError('use Python print/load for inspection')
        except ValueError as error:
            feedback=dict(accepted=False,error=str(error))
        else:
            candidate,feedback=compiler(source,current,turn,config);feedback['accepted']=False
            if feedback['exit_code']==0:
                try:
                    proposed=read_bank(candidate,reference);validate(proposed,reference,stage,frozen)
                except (ValueError,OSError,EOFError,zipfile.BadZipFile) as error:feedback['error']=str(error)
                else:
                    # Inference/infrastructure errors abort; they are not proposal rejections.
                    measured=evaluate(proposed,stage)
                    bank=proposed;np.savez_compressed(current,**bank)
                    feedback.update(accepted=True,target=measured)
        entry.update(status='accepted' if feedback.get('accepted') else 'rejected',feedback=feedback)
        save(turn/'feedback.json',feedback);history.append(dict(assistant=answer,feedback=feedback));snapshot(stage,step,turns)
    final=folder/'final.npz';np.savez_compressed(final,**bank)
    save(folder/'session.json',dict(stage=stage,turns=turns,sha256=sha(final),selection='last accepted'))
    return bank


def run(constructor,output_dir,config):
    import torch
    out=Path(output_dir).resolve();receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('source receipt mismatch')
    def verify_resources():
        for path,h in config['resources'].items():
            if sha(path)!=h:raise ValueError('resource changed: '+path)
    verify_resources()
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['runtime_verify']),
                    '--verify',linux_path(config['linux_identity'])],check=True,capture_output=True,timeout=90)
    started=time.monotonic();state=dict(status='loading',manifest_sha256=receipt['manifest_sha256'])
    def snapshot():state['active_seconds']=time.monotonic()-started;save_status(out/'status.json',state)
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def check():
        if stop():raise RuntimeError('stopped or wall guard')
    try:
        study=constructor();original=model_hash(study.model)
        if original!=config['model_sha256']:raise ValueError('unexpected base model')
        target=json.loads(Path(config['target']).read_text());reference,baseline=make_bank(study,target)
        np.savez_compressed(out/'reference.npz',**reference);save(out/'baseline_target.json',baseline)
        old=config['saved_target_score']
        if baseline['correct'] or baseline['prediction']!=old['prediction'] or not np.allclose(baseline['logprobs'],old['normalized_logprobs'],atol=2e-5,rtol=0):
            raise ValueError('target failure did not reproduce')
        noop=prefix_score(study,target,reference,'forced')
        if baseline!=noop:raise ValueError('disabled intervention differs')
        guide=Path(config['guide']).read_text(encoding='utf-8');threads=set()
        cli=json.loads(Path(config['local_paths']).read_text())['codex_cli']
        def collector(folder,text):
            check();t=time.monotonic();last=[0.]
            state.update(status='authoring',stage=folder.parent.name,step=int(folder.name.split('_')[1]));snapshot()
            def heartbeat():
                now=time.monotonic()
                if now-last[0]>5:last[0]=now;state.update(call=str(folder),call_active_seconds=now-t);snapshot()
                return stop()
            answer,audit=collect(cli,config['catalog'],config['instructions'],folder,text,1200,heartbeat)
            if audit['thread_id'] in threads:raise ValueError('reused thread')
            threads.add(audit['thread_id']);return answer,audit
        def evaluate(bank,stage):check();return prefix_score(study,target,bank,stage)
        def progress(stage,step,turns):state.update(status='authoring',stage=stage,step=step,turns=turns);snapshot()
        public=dict(target,baseline_score=baseline)
        if config.get('resume_root'):
            for path in (Path(config['resume_root'])/'forced').glob('turn_*/events.jsonl'):
                for line in path.read_text().splitlines():
                    event=json.loads(line)
                    if event.get('type')=='thread.started':threads.add(event['thread_id'])
        forced=session('forced',reference,reference,public,out/'forced',guide,config,collector,evaluate,construct,progress,resume=config.get('resume_root'))
        # A zero forced artifact is retained as a failure; retrieval is still reported.
        public['forced_stage_score']=evaluate(forced,'forced')
        retrieval=session('retrieval',forced,reference,public,out/'retrieval',guide,config,collector,evaluate,construct,progress,frozen=forced)
        frozen={name:sha(out/name/'final.npz') for name in ('forced','retrieval')};save(out/'frozen.json',frozen)
        # Never expose these questions or predictions to the author.
        panel=json.loads(Path(config['panel']).read_text());rows=[dict(target,category='target')]+panel
        evaluations={}
        variants=[('untouched',None,'off'),('forced',forced,'forced'),('retrieval',retrieval,'retrieval'),('zero_key',retrieval,'zero_key')]
        for name,bank,mode in variants:
            records=[];state.update(status='evaluating',variant=name);snapshot()
            for row in rows:
                check();record=prefix_score(study,row,bank,mode);record['category']=row['category'];records.append(record)
            save(out/(name+'_evaluation.json'),records);evaluations[name]=records
        if any(sha(out/name/'final.npz')!=h for name,h in frozen.items()):raise ValueError('frozen candidate changed')
        diagnostics={}
        for name,records in evaluations.items():
            by_id={r['id']:r for r in records};base={r['id']:r for r in evaluations['untouched']}
            categories=sorted({r['category'] for r in records})
            # Reversed-choice copies are a separate invariance check, not extra independent cases.
            originals=[r for r in records if not r['id'].endswith('_reversed')]
            parity=max(max(abs(a-b) for a,b in zip(r['logprobs'],reversed(by_id[r['id']+'_reversed']['logprobs']))) for r in originals)
            if parity!=0:raise ValueError('choice-order invariance failed')
            diagnostics[name]=dict(choice_order_max_error=parity,
                corrected=[r['id'] for r in originals if r['correct'] and not base[r['id']]['correct']],
                broken=[r['id'] for r in originals if not r['correct'] and base[r['id']]['correct']],
                categories={c:dict(correct=sum(r['correct'] for r in originals if r['category']==c),n=sum(r['category']==c for r in originals)) for c in categories})
        if model_hash(study.model)!=original:raise ValueError('model weights changed')
        verify_resources();state.update(status='complete',evaluations=evaluations,diagnostics=diagnostics,model_sha256=original,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
