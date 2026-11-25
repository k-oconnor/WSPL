"""Source-gated continuation; reuses audited prefix, performs no new control fit."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from skillforge.authoring_sft import restore_adapter,adapter_state,frozen_hash
from skillforge.teaching_training import state_hash
from skillforge.fresh_robot import sha
from skillforge.robot_trials import panel,acquire,evaluate,compare,CONDITIONS
from skillforge.robot_diagnostics import features
from skillforge.robot_gradient import fit
from skillforge.robot_authoring_access_resume import run_session
from skillforge.robot_access import inspect_acquisition
from skillforge.robot_submissions import load_arrays,freeze_ledger,verify_frozen
from skillforge.wspl_lora import identity as adapter_identity,describe
from skillforge.reliable_json import save,status
from skillforge.robot_recovery import load_prefix


ORDER=('native0','wspl0','wspl1','native1','native2','wspl2')


def validate_splits(config):
    groups=[panel(config[k],n) for k,n in [('acquisition_seed',4),('development_seed',4),('final_seed',16)]]
    seeds=[r['seed'] for g in groups for r in g]
    if config['final_seed']!=73600000:raise ValueError('Fresh reserved panel required')
    if len(seeds)!=len(set(seeds)) or any(73100000<=s<73100064 or 73200000<=s<73200016 for s in seeds):
        raise ValueError('Mission seeds overlap each other or parent panels')
    return groups


def summary(rows):
    return {c:dict(n=sum(r['condition']==c for r in rows),
                   successes=sum(bool(r['success']) for r in rows if r['condition']==c),
                   model_calls=sum(r['model_calls'] for r in rows if r['condition']==c),
                   invalid_tokens=sum(r['invalid_tokens'] for r in rows if r['condition']==c)) for c in CONDITIONS}


def run(constructor,output_dir,config):
    out=Path(output_dir);started=time.monotonic();compute_seconds=config['prior_compute_seconds'];final_started=None
    pin=json.loads((out/'source_receipt.json').read_text())['manifest_sha256']
    manifest=out/'approved_manifest.json';live={};model=None;parent=None
    def stopped():
        control=out/'control.json'
        if control.exists() and json.loads(control.read_text()).get('stop'):return True
        if final_started is None:return time.monotonic()-started>config['development_wall_seconds']
        return time.monotonic()-final_started>config['final_wall_seconds']
    def guard():
        if stopped():raise InterruptedError('Study stop/wall guard; no automatic retry')
        if final_started is None and compute_seconds>config['compute_seconds']:
            raise InterruptedError('Aggregate development compute guard')
    def charge(seconds):
        nonlocal compute_seconds
        compute_seconds+=seconds;guard()
    def report(**values):
        live.update(values,elapsed_seconds=time.monotonic()-started,development_compute_seconds=compute_seconds)
        status(out/'status.json',live)
    def install(bank):
        restore_adapter(model,{k:torch.from_numpy(a) for k,a in bank.items()})
        if adapter_identity({k:v.numpy() for k,v in adapter_state(model).items()})!=adapter_identity(bank):
            raise ValueError('Exact factor installation failed')
    def validation(candidate,folder):
        try:
            install(load_arrays(candidate))
            rows=evaluate(model,tokenizer,config,folder,manifest,pin,development,guard,report)
            return dict(summary=summary(rows),paired=compare(base_dev,rows)['conditions'])
        except Exception as error:
            raise RuntimeError('Development evaluator failure; no automatic retry') from error
        finally:restore_adapter(model,parent)
    try:
        report(status='verifying_resources')
        def check(item):
            path,expected=item
            if sha(path)!=expected:raise ValueError('Bound resource changed: '+path)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in pool.map(check,config['resources'].items()):guard()
        recovery=json.loads(Path(config['author_recovery_inventory']).read_text())
        save(out/'recovery_context.json',recovery)
        acquisition,development,reserved=validate_splits(config)
        # Merely constructing seed descriptors is not opening/scoring final missions.
        study=constructor();model,tokenizer=study.model,study.tokenizer
        saved=torch.load(config['parent_checkpoint'],map_location='cpu',weights_only=True)
        parent=saved['adapter'];base_hash=frozen_hash(model)
        if base_hash!=config['base_hash'] or state_hash(parent)!=config['parent_factor_hash']:
            raise ValueError('Frozen parent/base identity mismatch')
        restore_adapter(model,parent)
        if state_hash(adapter_state(model))!=config['parent_factor_hash']:raise ValueError('Parent restore failed')
        report(status='verifying_reused_prefix');t=time.monotonic()
        common,rows,catalog,base_dev,gradient=load_prefix(config,out,guard)
        if adapter_identity(load_arrays(common/'parent.npz'))!=adapter_identity({k:v.numpy() for k,v in parent.items()}):
            raise ValueError('Recovered common parent differs from deployed parent')
        if summary(base_dev)!=json.loads((common/'parent_development.json').read_text()):
            raise ValueError('Recovered development summary mismatch')
        charge(time.monotonic()-t)
        candidates={'parent':common/'parent.npz','gradient':gradient}
        # Present a neutral all-projection catalog, not the old last-layer feature pack.
        (common/'features.npz').unlink()
        catalog.pop('diagnostic_projection',None);catalog.pop('feature_shapes',None)
        save(common/'catalog.json',catalog)
        save(out/'common_input_hashes.json',{p.name:sha(p) for p in common.iterdir() if p.is_file()})
        packet=dict(adapter_id=catalog['adapter_id'],projection_count=len(catalog['projections']),
                    element_count=catalog['element_count'],projections=list(catalog['projections']),
                    acquisition_rows=len(rows),parent_development=summary(base_dev))
        def query(request,candidate,folder):
            guard()
            bank=load_arrays(candidate) if candidate is not None else {k:v.numpy() for k,v in parent.items()}
            return inspect_acquisition(model,tokenizer,rows,bank,request,folder,guard,
                max_elements=config['inspection_elements'],max_seconds=config['inspection_seconds'])
        for name in ORDER:
            guard();restore_adapter(model,parent)
            if name in recovery['completed_authors']:
                item=recovery['candidates'][name]
                if sha(item['file'])!=item['sha256'] or adapter_identity(load_arrays(item['file']))!=item['adapter_id']:
                    raise ValueError('Completed candidate changed')
                candidates[name]=Path(item['file'])
                continue
            resume=recovery['resume'] if name==recovery['resume_author'] else None
            candidates[name]=run_session(name.rstrip('012'),common,packet,config,out/name,manifest,pin,
                guard,stopped,report,validation,charge,query_candidate=query,resume=resume)
            if frozen_hash(model)!=base_hash:raise ValueError('Base identity changed after author session')
            for file,expected in json.loads((out/'common_input_hashes.json').read_text()).items():
                if sha(common/file)!=expected:raise ValueError('Common author input changed')
        ledger=freeze_ledger(candidates,out/'all_frozen.json')
        # No author or optimizer can run after this boundary. Final simulator opens here.
        final_started=time.monotonic();predictions={};comparisons={}
        for name in ('parent','gradient',*ORDER):
            guard();entry=ledger[name]
            if entry['status']=='authoring_failed':continue
            install(verify_frozen(entry));report(status='final_evaluation',final_arm=name)
            predictions[name]=evaluate(model,tokenizer,config,out/('final_'+name),manifest,pin,reserved,guard,report)
            if name!='parent':comparisons[name]=compare(predictions['parent'],predictions[name])
        for entry in ledger.values():
            if entry['status']=='frozen':verify_frozen(entry)
        restore_adapter(model,parent)
        if frozen_hash(model)!=base_hash or state_hash(adapter_state(model))!=config['parent_factor_hash']:
            raise ValueError('Final base/parent restoration failed')
        usage=[]
        for name in ORDER:
            for file in sorted((out/name).glob('turn_*/usage.json')):
                usage.append(dict(author=name,**json.loads(file.read_text())))
        result=dict(status='complete',manifest_sha256=pin,elapsed_seconds=time.monotonic()-started,
            development_compute_seconds=compute_seconds,acquisition_missions=16,acquisition_decisions=len(rows),
            final_missions_per_valid_arm=64,author_order=ORDER,arms=ledger,comparisons=comparisons,
            author_calls=len(usage),usage=usage,method_classification='pending_source_audit',
            study='broader_access_recovery',historical_control_reused=True,new_gradient_updates=0,
            prior_completed_author_responses=recovery['prior_completed_responses'],prior_failed_author_invocations=1,
            prior_failed_author_usage='unknown',prior_elapsed_seconds=recovery['prior_elapsed_seconds'],
            fresh_author_calls=len(usage),
            total_completed_author_responses=recovery['prior_completed_responses']+len(usage),
            reused_gradient_updates=384,inspection_attempts=sum(json.loads(((Path(recovery['prior_output']) if n in recovery['completed_authors'] else out)/n/'frozen.json').read_text())['inspection_attempts'] for n in ORDER),
            failure_adjusted_gate_passes={arm:sum(bool(comparisons.get(arm+str(i),{}).get('exploratory_gate')) for i in range(3))
                                        for arm in ('native','wspl')},follow_on_launched=False)
        save(out/'results.json',result);report(status='complete')
    except BaseException as error:
        report(status='failed',error=repr(error));raise
    finally:
        if model is not None and parent is not None:restore_adapter(model,parent)


if __name__=='__main__':
    from source_consistency import require_registered_runner
    require_registered_runner(__file__)
