"""One six-response native key session with frozen value and new evaluation panel."""
import json
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from skillforge.arc_authoring import read_bank
from skillforge.arc_eval import model_hash
from skillforge.attention_probe import prefix_score
from skillforge.selective_key import features,validate_key,evaluate,descriptive_gate
from skillforge.reliable_json import save,status as save_status
from skillforge.codex_one_shot import collect
from skillforge.diffusion_edit import linux_path
from skillforge.go_replay import sha
from run_attention_probe import session,construct


def run(constructor,output_dir,config):
    import torch
    out=Path(output_dir).resolve();receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('source receipt required')
    def verify_resources():
        for p,h in config['resources'].items():
            if sha(p)!=h:raise ValueError('changed resource: '+p)
    verify_resources()
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['runtime_verify']),'--verify',linux_path(config['linux_identity'])],check=True,capture_output=True,timeout=90)
    started=time.monotonic();state=dict(status='loading',manifest_sha256=receipt['manifest_sha256'])
    def snapshot():state['active_seconds']=time.monotonic()-started;save_status(out/'status.json',state)
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def check():
        if stop():raise RuntimeError('stop/guard')
    try:
        study=constructor();before=model_hash(study.model)
        if before!=config['model_sha256']:raise ValueError('model changed')
        # Parent candidate is host-owned, previously validated and pinned; no author file loaded here.
        with np.load(config['parent_bank'],allow_pickle=False) as z:parent={k:z[k].copy() for k in z.files}
        active=np.argwhere(parent['alpha']>0)
        if active.tolist()!=[[2,0]] or parent['alpha'][2,0]!=.75:raise ValueError('unexpected parent slot')
        training=json.loads(Path(config['training']).read_text())
        q,logz,base=features(study,training)
        reference={**parent,'train_Q':q,'train_logZ':logz,
            'train_applicable':np.asarray([r['answerKey']==r['choices']['label'][2] for r in training],np.float32)}
        save(out/'training_baseline.json',base);np.savez_compressed(out/'reference.npz',**reference)
        noop=evaluate(study,training,reference,'off',base,check)
        if noop['records']!=base:raise ValueError('disabled injection mismatch')
        old=json.loads(Path(config['parent_results']).read_text())
        original=evaluate(study,training,parent,'retrieval',base,check)
        stored={x['id']:x for x in old['evaluations']['retrieval']}
        for row in original['records']:
            # Archived rows have an extra category; inference fields must match exactly.
            if row!={k:v for k,v in stored[row['id']].items() if k!='category'}:raise ValueError('parent replay differs')
        save(out/'training_parent.json',original)
        guide=Path(config['guide']).read_text();packet=dict(examples=training,baseline=base,starting_key=original,
            note='These are all acquisition examples, including the previous evaluation panel. Only K[2,0,:] may change.')
        threads=set();cli=json.loads(Path(config['local_paths']).read_text())['codex_cli']
        def collector(folder,text):
            check();t=time.monotonic();last=[0.];state.update(status='authoring',step=int(folder.name.split('_')[1]));snapshot()
            def heartbeat():
                now=time.monotonic()
                if now-last[0]>5:last[0]=now;state.update(call_active_seconds=now-t);snapshot()
                return stop()
            answer,audit=collect(cli,config['catalog'],config['instructions'],folder,text,1200,heartbeat)
            if audit['thread_id'] in threads:raise ValueError('thread reused')
            threads.add(audit['thread_id']);return answer,audit
        def assess(bank,stage):
            validate_key(bank,reference)
            return evaluate(study,training,bank,'retrieval',base,check)
        def compiler(source,current,turn,cfg):
            path,feedback=construct(source,current,turn,cfg)
            if feedback['exit_code']==0:
                try:validate_key(read_bank(path,reference),reference)
                except (ValueError,OSError,EOFError,zipfile.BadZipFile) as e:feedback.update(exit_code=1,stderr=str(e))
            return path,feedback
        def progress(stage,step,turns):state.update(status='authoring',step=step,turns=turns);snapshot()
        final=session('retrieval',reference,reference,packet,out/'key',guide,config,collector,assess,compiler,progress,frozen=reference)
        frozen=sha(out/'key/final.npz');save(out/'frozen.json',dict(sha256=frozen))
        # New cases are inaccessible to author and opened only after freeze.
        panel=json.loads(Path(config['panel']).read_text());state.update(status='evaluating');snapshot()
        panel_base=[]
        for row in panel:check();panel_base.append(prefix_score(study,row))
        reports={}
        for name,bank,mode in [('untouched',None,'off'),('forced',parent,'forced'),('old_key',parent,'retrieval'),('new_key',final,'retrieval'),('zero_key',final,'zero_key')]:
            state.update(variant=name);snapshot()
            reports[name]=evaluate(study,panel,bank,mode,panel_base,check);save(out/(name+'_evaluation.json'),reports[name])
        if sha(out/'key/final.npz')!=frozen or model_hash(study.model)!=before:raise ValueError('frozen artifact/model changed')
        verify_resources();state.update(status='complete',reports=reports,descriptive_criteria=descriptive_gate(reports['new_key'],reports['untouched']),
            model_sha256=before,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
