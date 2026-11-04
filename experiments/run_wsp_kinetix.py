"""Approved fresh Astra Kinetix repair screen; Windows collection, WSL evaluation."""
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from skillforge.codex_one_shot import collect
from skillforge.kinetix_replay import sha,ALIASES
from skillforge.wsp_kinetix import prompt
from run_kinetix_replay import save

ORDER=((0,'native'),(0,'wspl'),(1,'wspl'),(1,'native'),(2,'native'),(2,'wspl'))


def linux(path):
    p=Path(path).resolve()
    if p.drive.lower()!='c:':raise ValueError('C drive required')
    return '/mnt/c/'+p.as_posix()[3:]


def comparison(trials):
    means={}
    for arm in ('native','wspl'):
        rows=[r for r in trials if r['arm']==arm]
        if len(rows)!=3 or any(r['status'] not in ('complete','invalid') for r in rows):return dict(decision='incomplete')
        means[arm]=float(np.mean([r['evaluation']['success_rate'] if r['status']=='complete' else 0 for r in rows]))
    decision='unresolved_native_floor' if means['native']<.5 else ('feasible' if means['wspl']>=.5 and means['wspl']>=.9*means['native'] and all(r['status']=='complete' for r in trials if r['arm']=='wspl') else 'no_go_this_configuration')
    return dict(failure_adjusted_mean_success=means,decision=decision,scope='descriptive, three author draws per arm; not statistical noninferiority')


def run(constructor,output_dir,config):
    out=Path(output_dir).resolve();root=Path.cwd();started=time.monotonic()
    receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or receipt['manifest_sha256']!=sha(out/'approved_manifest.json'):raise ValueError('source receipt required')
    def verify_inputs():
        for path,digest in config['bound_files'].items():
            if sha(path)!=digest:raise ValueError('bound file differs: '+path)
    verify_inputs()
    state=dict(status='running',trials=[],planned_invocations=6,training_updates=0,local_gpu_used=False,formal_status='UNSUPPORTED_TANH')
    def snapshot():
        state.update(active_seconds=time.monotonic()-started,manifest_sha256=receipt['manifest_sha256'])
        save(out/'status.json',state)
    def stop():
        control=out/'control.json'
        return time.monotonic()-started>21600 or (control.exists() and json.loads(control.read_text()).get('stop'))
    def evaluate(folder):
        args=['wsl.exe','-d','Ubuntu','--','env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','JAX_PLATFORMS=cpu',config['linux_python'],linux(root/'experiments/kinetix_author_eval.py'),'--run',linux(out),'--folder',linux(folder),'--digest',receipt['manifest_sha256'],'--weights-sha256',sha(folder/'policy.npz')]
        save(folder/'evaluation_command.json',args)
        with (folder/'evaluation.stdout.log').open('wb') as stdout,(folder/'evaluation.stderr.log').open('wb') as stderr:
            proc=subprocess.Popen(args,stdout=stdout,stderr=stderr)
            while proc.poll() is None:
                if stop():save(out/'control.json',dict(stop=True,reason='user_stop_or_wall_guard'))
                snapshot();time.sleep(2)
            if proc.returncode:raise RuntimeError('evaluation worker failed; no retry')
        return json.loads((folder/'evaluation.json').read_text())
    try:
        parent=constructor();folder=out/'parent_control';folder.mkdir()
        parent.parent.save(folder/'parent_artifact.json')
        np.savez_compressed(folder/'policy.npz',**parent.original)
        state['parent_control']=evaluate(folder);snapshot()
        if state['parent_control']['status']!='complete':raise RuntimeError('parent panel incomplete')
        if state['parent_control']['success_rate']>.1:
            state.update(status='complete',comparison=dict(decision='unresolved_parent_no_longer_failing'),authoring_skipped=True)
            verify_inputs();snapshot();save(out/'results.json',state);return
        threads=set()
        for draw,arm in ORDER:
            if stop():state['status']='paused';break
            trial=dict(draw=draw,arm=arm,status='collecting');state['trials'].append(trial);snapshot()
            policy=constructor();folder=out/f'draw_{draw}_{arm}'
            text=Path(config['prompts'][arm]).read_text(encoding='utf-8')
            expected=prompt(Path(config['reference_prompt']).read_text(encoding='utf-8'),Path(config['guide']).read_text(encoding='utf-8'),arm,policy.parent.sha256)
            if text!=expected:raise ValueError('prompt differs')
            answer,audit=collect(config['cli'],config['catalog'],config['instructions'],folder,text,1200,stop)
            if audit['thread_id'] in threads:raise ValueError('fresh thread required')
            threads.add(audit['thread_id']);trial['collection']=audit
            try:artifact,plan,source=policy.construct(arm,answer)
            except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,SyntaxError,RecursionError) as exc:
                trial.update(status='invalid',error=f'{type(exc).__name__}: {exc}');save(folder/'invalid.json',trial);snapshot();continue
            artifact.save(folder/'candidate.json');plan.save(folder/'edit.plan.json')
            (folder/'source.wspl').write_text(source,encoding='utf-8')
            np.savez_compressed(folder/'policy.npz',**policy.weights)
            trial.update(candidate_id=artifact.sha256,source_bytes=len(source.encode()),numeric_profile=plan.to_dict()['numeric_profile'],resources=plan.to_dict()['resources'],changed_parameters=sum(int(np.count_nonzero(policy.weights[k]!=policy.original[k])) for k in policy.weights))
            # Frozen host adapter parity; no simulator evaluation in this check.
            for row in np.random.default_rng(814).uniform(-10,10,(8,679)).astype(np.float32):
                assert np.array_equal(policy.logits(row).view('u4'),np.asarray(artifact.trace(row,semantics='numpy_f32')['y'],dtype=np.float32).view('u4'))
            trial['status']='evaluating';snapshot();trial['evaluation']=evaluate(folder)
            trial['status']='complete' if trial['evaluation']['status']=='complete' else 'incomplete';snapshot()
            if trial['status']!='complete':state['status']='paused';break
        if state['status']=='running':state['status']='complete'
        state['comparison']=comparison(state['trials']);verify_inputs();snapshot();save(out/'results.json',state)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');snapshot();save(out/'results.json',state);raise
