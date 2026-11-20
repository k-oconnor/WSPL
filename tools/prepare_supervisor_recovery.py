"""Prepare a source-bound recovery; no model calls or research execution."""
import copy,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import source_consistency as sc
from skillforge.supervisor_workspace import digest,save
from skillforge.supervisor_recovery import restore
from skillforge.codex_one_shot import preflight
from skillforge.supervisor_session import TOOL_INSTRUCTIONS


def main():
    old=ROOT.parent/'WSPL-astra-supervisor-v3'
    m=sc.verify(old,old/'experiments/source_pins/astra_supervisor_v3.manifest.json',
        '7fe011f662c757695e2f9e497ab72727d53f9d8ed72c29b262d95c3e42a9f780')
    c=copy.deepcopy(m['runner_config']);parent=old/'results/astra_supervisor_v3'
    c['recovery_parent']=str(parent)
    c['readonly']['code']=str(ROOT/'src')
    for key in ('runtime_verifier','template','job_worker','final_worker','deployment_worker'):
        c[key]=str(ROOT/'tools'/Path(c[key]).name)
    parent_files=[p for p in parent.rglob('*') if p.is_file()]
    for p in parent_files:c['resources'][str(p.resolve())]=digest(p)
    doc=ROOT/'docs/wspl/ASTRA_SUPERVISOR_RECOVERY_V1.md'
    c['resources'][str(doc)]=digest(doc)
    for p,h in c['resources'].items():
        if digest(p)!=h:raise ValueError('Changed resource '+p)
    brief=Path(c['brief']).read_text(encoding='utf8')+'\n'+c['workbench_help']
    data=ROOT/'data/recovery_preflight';data.mkdir(parents=True)
    with tempfile.TemporaryDirectory(dir=data) as t:
        restore(parent,Path(t),brief,c['limits'])
        receipt=json.loads((Path(t)/'recovery_receipt.json').read_text())
    state=json.loads((parent/'supervisor/session.json').read_text())
    prompt=brief+'\n'+TOOL_INSTRUCTIONS+'\nComplete visible history:\n'+json.dumps(state['transcript'])
    preview=data/'prompt';preview.mkdir();(preview/'empty').mkdir()
    preflight(c['cli'],c['catalog'],c['instructions'],preview,prompt,profile='supervisor')
    spec={k:copy.deepcopy(m[k]) for k in ('source_roots','constructor','constructor_kwargs','runner')}
    spec.update(model_id='astra-supervisor-resume-v1',runner_config=c)
    save(ROOT/'experiments/source_pins/astra_supervisor_resume_v1.spec.json',spec)
    receipt.update(parent_files=len(parent_files),resources=len(c['resources']),model_calls=0,offline_prompt='passed')
    save(ROOT/'docs/wspl/supervisor_recovery_preflight.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
