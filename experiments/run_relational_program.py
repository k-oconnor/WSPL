"""Bounded relational WSP+ASP test with one fixed native program artifact."""
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
from run_joint_attention import session
from skillforge.relational_program import NeuralProgram,fit_bias,apply_bias,pair_summary,decision
from skillforge.relational_program import make_program_reference as make_reference


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
        bias=fit_bias(base,acquired);save(out/'frozen_answer_bias.json',bias)
        # Expose actual acquisition token strings, avoiding token-ID guessing.
        decoded={str(int(i)):study.tokenizer.decode([int(i)]) for i in reference['answer_token_ids']}
        packet=dict(examples=acquired,baseline=base,initial_asp=initial,answer_token_strings=decoded,tensors={k:list(v.shape) for k,v in reference.items()})
        guide=Path(config['guide']).read_text(encoding='utf-8');cli=json.loads(Path(config['local_paths']).read_text())['codex_cli'];threads=set()
        def collector(folder,text):
            check();state.update(status='authoring',turn=folder.name);snapshot();t=time.monotonic()
            def heartbeat():state['call_active_seconds']=time.monotonic()-t;snapshot();return stop()
            answer,audit=collect(cli,config['catalog'],config['instructions'],folder,text,1200,heartbeat)
            if audit['thread_id'] in threads:raise ValueError('reused thread')
            threads.add(audit['thread_id']);return answer,audit
        def evaluate(bank):
            records=[]
            program=NeuralProgram(bank)
            for row in acquired:check();records.append(program.score(study,row))
            if model_hash(study.model)!=before:raise ValueError('base restoration')
            return dict(records=records,metrics=metrics(records,base,acquired))
        def checkpoint(step,turns):state.update(step=step,turns=turns);snapshot()
        candidate=session(reference,out/'author',packet,guide,collector,lambda s,p,t:construct(s,p,t,config),evaluate,checkpoint)
        frozen=sha(out/'author/final.npz');save(out/'frozen.json',dict(sha256=frozen,selection='last accepted'))
        compact={k:candidate[k].copy() for k in ('K','V','alpha','W27')}
        np.savez_compressed(out/'program.npz',**compact);program_hash=sha(out/'program.npz')
        save(out/'program.json',dict(schema=1,bank='program.npz',bank_sha256=program_hash,
             base_model_sha256=before,source_sha256=receipt['manifest_sha256'],
             hierarchy=[dict(operation='virtual_attention_slot',layer=26,kv_group=0,position='last_question_prefix_query'),
                        dict(operation='replace_attention_output_weight',layer=27,positions='all')],
             execution='NeuralProgram.score: one fixed forward computation per candidate continuation; no external router or task solver'))
        # Reserved examples are first opened only after freezing the final artifact.
        rows=json.loads(Path(config['reserved']).read_text());outputs={};summary={}
        program=NeuralProgram(compact)
        for name in ('untouched','joint','wsp_only','asp_only'):
            state.update(status='evaluating',variant=name);snapshot();records=[]
            for row in rows:check();records.append(program.score(study,row,name))
            outputs[name]=records;summary[name]=metrics(records,outputs['untouched'],rows);save(out/(name+'_reserved.json'),records)
        outputs['answer_bias']=apply_bias(outputs['untouched'],rows,bias)
        summary['answer_bias']=metrics(outputs['answer_bias'],outputs['untouched'],rows)
        save(out/'answer_bias_reserved.json',outputs['answer_bias'])
        pairs={name:pair_summary(records,rows) for name,records in outputs.items()}
        verdict=decision(summary,pairs,outputs,rows)
        state.update(pair_summary=pairs,decision=verdict)
        if sha(out/'author/final.npz')!=frozen or sha(out/'program.npz')!=program_hash or model_hash(study.model)!=before:raise ValueError('frozen identity')
        resources();state.update(status='complete',summary=summary,final_sha256=frozen,model_sha256=before)
    except BaseException as e:state.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:snapshot();save(out/'results.json',state)
