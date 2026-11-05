"""Matched bounded construction sessions, then frozen Go evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
from skillforge.go_authoring import MAX_STEPS,conversation_prompt,reply_action,state_summary
from skillforge.diffusion_edit import linux_path
from skillforge.go_replay import sha
from skillforge.codex_one_shot import collect
from run_diffusion_replay import save

ORDER=[(d,a) for d in range(3) for a in (('python','wspl') if d%2==0 else ('wspl','python'))]


def import_frozen(out, config):
    """Narrow recovery: only a frozen prefix followed by an unanswered first call."""
    recovery=config.get('resume')
    if not recovery:return [],set(),{}
    parent=Path(recovery['parent'])
    for relative,digest in recovery['files'].items():
        path=parent/relative
        if not path.resolve().is_relative_to(parent.resolve()) or sha(path)!=digest:
            raise ValueError('resume archive changed: '+relative)
    state=json.loads((parent/'status.json').read_text())
    rows=state['trials'];prefix=rows[:-1]
    if (state['status']!='incomplete' or state['error']!='RuntimeError: call_wall_guard'
        or len(prefix)!=4 or rows[-1]['name']!='2_python' or rows[-1]['turns']):
        raise ValueError('unsupported recovery state')
    frozen={};threads=set()
    for row,(draw,arm) in zip(prefix,ORDER):
        name=f'{draw}_{arm}'
        if row['name']!=name or row['status']!='frozen' or sha(parent/name/'final.npz')!=row['final_sha256']:
            raise ValueError('invalid frozen prefix')
        shutil.copytree(parent/name,out/name)
        frozen[name]=dict(path=name+'/final.npz',sha256=row['final_sha256'])
        for turn in row['turns']:
            thread=turn['collection']['thread_id']
            if thread in threads:raise ValueError('duplicate parent thread')
            threads.add(thread)
    save(out/'resume_provenance.json',dict(parent=str(parent),files=recovery['files'],
        parent_active_seconds=state['active_seconds'],interrupted_call='2_python/turn_00',
        interrupted_usage='unknown',imported_candidates=list(frozen)))
    return prefix,threads,frozen


def construct(arm,source,parent,folder,config):
    path=folder/('proposal.py' if arm=='python' else 'proposal.wspl');path.write_text(source,encoding='utf-8')
    receipt=folder/'construction.json';output=folder/'constructed'
    if arm=='python':
        command=['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['python_worker']),
            '--source',linux_path(path),'--parent',linux_path(parent),'--output',linux_path(output),
            '--runtime',config['python_runtime'],'--receipt',linux_path(receipt)]
    else:
        command=[sys.executable,config['wspl_worker'],'--source',str(path),'--parent',str(parent),'--output',str(output),'--receipt',str(receipt)]
    try:
        run=subprocess.run(command,capture_output=True,text=True,timeout=160)
    except subprocess.TimeoutExpired:
        return None,dict(exit_code=1,stdout='',stderr='construction wall limit exceeded; no state accepted')
    if not receipt.exists():raise RuntimeError('construction infrastructure failure: '+run.stderr[-1000:])
    feedback=json.loads(receipt.read_text())
    return output/'candidate.npz',feedback


def author_sessions(constructor,out,guide,config,collector,compiler,stop,snapshot):
    trials,threads,frozen=import_frozen(out,config)
    snapshot(trials)
    for draw,arm in ORDER[len(trials):]:
        name=f'{draw}_{arm}';folder=out/name;folder.mkdir();study=constructor();state=folder/'current.npz';study.save(state)
        history=[];row=dict(name=name,draw=draw,arm=arm,status='authoring',turns=[]);trials.append(row);snapshot(trials)
        for step in range(MAX_STEPS):
            if stop():raise RuntimeError('user stop or overall guard')
            text=conversation_prompt(guide,arm,history,step)
            turn=folder/f'turn_{step:02d}';answer,audit=collector(turn,text,stop)
            if audit['thread_id'] in threads:raise ValueError('fresh-call thread reused')
            threads.add(audit['thread_id']);entry=dict(step=step,collection=audit);row['turns'].append(entry)
            try:action,source=reply_action(answer,arm)
            except ValueError as e:
                action='invalid';feedback=dict(accepted=False,error=str(e))
            if action=='final':entry['status']='final';row['termination']='author_final';break
            if action=='construct':
                candidate,feedback=compiler(arm,source,state,turn,config)
                feedback['accepted']=False
                if feedback['exit_code']==0 and candidate is not None:
                    try:study.accept(candidate);study.save(state);feedback['accepted']=True
                    except (ValueError,OSError) as e:feedback['stderr']=str(e)
            feedback['weights']=state_summary(study.arrays)
            entry.update(status='accepted' if feedback['accepted'] else 'rejected',feedback=feedback)
            save(turn/'feedback.json',feedback);history.append(dict(assistant=answer,construction_feedback=feedback));snapshot(trials)
        else:row['termination']='construction_budget_exhausted'
        final=folder/'final.npz';study.save(final);row.update(status='frozen',final_sha256=sha(final))
        frozen[name]=dict(path=final.relative_to(out).as_posix(),sha256=sha(final));snapshot(trials)
    control=out/'zero_control';control.mkdir();constructor().save(control/'final.npz')
    frozen['zero_control']=dict(path='zero_control/final.npz',sha256=sha(control/'final.npz'))
    save(out/'frozen_candidates.json',frozen)
    return trials,frozen


def run(constructor,output_dir,config):
    out=Path(output_dir).resolve();receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('approved receipt required')
    for p,h in config['windows_resources'].items():
        if sha(p)!=h:raise ValueError('Windows resource differs: '+p)
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['runtime_verify']),'--verify',linux_path(config['linux_identity'])],check=True,capture_output=True,timeout=60)
    cli=json.loads(Path(config['local_paths']).read_text())['codex_cli'];guide=Path(config['guide']).read_text(encoding='utf-8');started=time.monotonic();state={}
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def snapshot(trials):
        state.update(status='authoring',trials=trials,active_seconds=time.monotonic()-started,manifest_sha256=receipt['manifest_sha256']);save(out/'status.json',state)
    def collector(folder,prompt,should_stop):
        call_started=time.monotonic();last=[0.0]
        def heartbeat():
            now=time.monotonic()
            if now-last[0]>=5:
                last[0]=now
                state.update(active_seconds=now-started,current_call=str(folder.relative_to(out)),call_active_seconds=now-call_started)
                save(out/'status.json',state)
            return should_stop()
        try:return collect(cli,config['catalog'],config['instructions'],folder,prompt,1200,heartbeat)
        finally:
            state.pop('current_call',None);state.pop('call_active_seconds',None)
    try:
        trials,frozen=author_sessions(constructor,out,guide,config,collector,construct,stop,snapshot)
        state['status']='evaluating';save(out/'status.json',state)
        for name in frozen:
            if stop():raise RuntimeError('user stop or overall guard')
            state['evaluating']=name;save(out/'status.json',state)
            cmd=['wsl','-d','Ubuntu','--','env','JAX_PLATFORMS=cpu','JAX_ENABLE_X64=false','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
                config['go_python'],linux_path(config['eval_worker']),'--study-output',linux_path(out),'--candidate',name]
            with (out/(name+'.eval.stdout')).open('wb') as stdout,(out/(name+'.eval.stderr')).open('wb') as stderr:
                proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr)
                while proc.poll() is None:
                    if stop():
                        save(out/'control.json',dict(stop=True));proc.wait(timeout=60);raise RuntimeError('evaluation stopped')
                    time.sleep(.5)
                if proc.returncode:raise RuntimeError('evaluation worker failed: '+name)
        state.update(status='complete',evaluations={name:json.loads((out/name/'evaluation/results.json').read_text()) for name in frozen},active_seconds=time.monotonic()-started)
        for p,h in config['windows_resources'].items():
            if sha(p)!=h:raise ValueError('resource changed: '+p)
    except BaseException as e:
        state.update(status='incomplete',error=f'{type(e).__name__}: {e}',active_seconds=time.monotonic()-started);raise
    finally:save(out/'status.json',state);save(out/'results.json',state)
