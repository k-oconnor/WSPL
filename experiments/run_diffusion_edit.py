"""Approved six-proposal diffusion coordinate-edit comparison."""
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from skillforge.diffusion_edit import ARMS,evaluate_arrays,linux_path
from skillforge.diffusion_replay import sha,contact_sheet_pgm
from skillforge.codex_one_shot import collect
from run_diffusion_replay import save

ORDER=[(d,a) for d in range(3) for a in (ARMS if d%2==0 else ARMS[::-1])]


def execute(constructor,out,config,collector,stop,snapshot):
    rows=[dict(draw=d,arm=a,status='not_attempted') for d,a in ORDER];threads=set()
    state=dict(status='running',trials=rows);snapshot(state)
    for row in rows:
        if stop():state['status']='paused';break
        study=constructor();folder=out/f"{row['draw']}_{row['arm']}"
        prompt=Path(config['prompts'][row['arm']]).read_text(encoding='utf-8')
        row['status']='collecting';snapshot(state)
        try:
            answer,audit=collector(folder,prompt,stop)
            row['collection']=audit
            if audit['thread_id'] in threads:raise ValueError('thread reused')
            threads.add(audit['thread_id'])
        except Exception as e:
            row.update(status='collection_failed',error=f'{type(e).__name__}: {e}');state['status']='incomplete';break
        row['status']='constructing';snapshot(state);started=time.monotonic()
        for name,parent in study.parents.items():
            np.savez_compressed(folder/(name+'_parent.npz'),**parent)
        try:
            candidates,receipts=study.construct(row['arm'],answer,folder,config['worker'],config['linux_runtime'])
        except (ValueError,TypeError,KeyError,IndexError,ArithmeticError,SyntaxError,RecursionError) as e:
            row.update(status='invalid_proposal',error=f'{type(e).__name__}: {e}',construction_seconds=time.monotonic()-started)
            snapshot(state);continue
        row['construction_seconds']=time.monotonic()-started;save(folder/'construction.json',receipts)
        row['status']='evaluating';snapshot(state);evaluations={}
        for name,candidate in candidates.items():
            try:
                evaluation,samples=evaluate_arrays(study.parents[name],candidate)
                np.savez_compressed(folder/(name+'_paired_samples.npz'),**samples)
                for k in ('parent','candidate','target'):contact_sheet_pgm(samples[k],folder/(name+'_'+k+'.pgm'))
            except (ValueError,ArithmeticError) as e:
                evaluation=dict(numeric_pass=False,structural_certificate='NOT_ESTABLISHED',error=f'{type(e).__name__}: {e}')
            evaluations[name]=evaluation
        save(folder/'evaluation.json',evaluations)
        row.update(status='evaluated',evaluation=evaluations,numeric_success=all(e['numeric_pass'] for e in evaluations.values()),
            structural_certificates=sum(e['structural_certificate']=='PROVED_REAL_GRAPH_PERMUTATION' for e in evaluations.values()),
            candidate_hashes={n:sha(folder/(n+'_candidate.npz')) for n in candidates})
        snapshot(state)
    if state['status']=='running':state['status']='complete'
    state['summary']={a:dict(planned=3,evaluated=sum(r['status']=='evaluated' for r in rows if r['arm']==a),
        successful=sum(r.get('numeric_success',False) for r in rows if r['arm']==a),
        output_tokens=sum(r.get('collection',{}).get('usage',{}).get('output_tokens',0) for r in rows if r['arm']==a)) for a in ARMS}
    snapshot(state);return state


def run(constructor,output_dir,config):
    out=Path(output_dir);receipt=json.loads((out/'source_receipt.json').read_text())
    if receipt['status']!='verified' or sha(out/'approved_manifest.json')!=receipt['manifest_sha256']:raise ValueError('approved receipt required')
    if config['call_guard_seconds']!=1200 or config['wall_guard_seconds']!=14400:raise ValueError('usage guards differ')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('single-thread CPU runtime required')
    def verify():
        for p,digest in config['resources'].items():
            if sha(p)!=digest:raise ValueError('resource changed: '+p)
        subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(config['runtime_tool']),
            '--verify',linux_path(config['linux_identity'])],check=True,capture_output=True,timeout=60)
    verify();cli=json.loads(Path(config['local_paths']).read_text())['codex_cli'];started=time.monotonic()
    def stop():
        p=out/'control.json'
        return time.monotonic()-started>config['wall_guard_seconds'] or (p.exists() and json.loads(p.read_text()).get('stop'))
    def snapshot(state):
        state.update(manifest_sha256=receipt['manifest_sha256'],active_seconds=time.monotonic()-started);save(out/'status.json',state)
    def collector(folder,prompt,should_stop):
        return collect(cli,config['catalog'],config['instructions'],folder,prompt,config['call_guard_seconds'],should_stop)
    final=execute(constructor,out,config,collector,stop,snapshot)
    verify();save(out/'results.json',final)
