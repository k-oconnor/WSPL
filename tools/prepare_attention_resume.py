"""Pin a recovery of exactly two completed native-attention turns; no model calls."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from prepare_arc_baseline import sha
from source_consistency import inventory,write_json,verify
from skillforge.arc_authoring import read_bank
from skillforge.attention_probe import validate,author_prompt
from skillforge.codex_one_shot import audit_events,preflight


def main():
    import numpy as np
    oldroot=ROOT.parent/'Skillforge-attention-probe-v1';parent=oldroot/'results/attention_probe_v1'
    pin='bd1ed8f854222d6f4950aa5df1f1cd914b9c8890985a651f2b9c513b01c43480'
    old=verify(oldroot,oldroot/'experiments/source_pins/attention_probe_v1.manifest.json',pin)
    state=json.loads((parent/'status.json').read_text());assert state['status']=='incomplete' and state['step']==1
    assert 'PermissionError' in state['error'] and 'status.json' in state['error']
    with np.load(parent/'reference.npz',allow_pickle=False) as z:reference={k:z[k].copy() for k in z.files}
    history=[]
    for step in range(2):
        folder=parent/'forced'/f'turn_{step:02d}';answer=(folder/'answer.md').read_text(encoding='utf-8')
        audit_events([json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()],answer)
        feedback=json.loads((folder/'feedback.json').read_text());assert feedback['accepted']
        bank=read_bank(folder/'constructed/candidate.npz',reference);validate(bank,reference,'forced')
        history.append(dict(assistant=answer,feedback=feedback))
    current=read_bank(parent/'forced/current.npz',reference)
    assert all(current[k].tobytes()==bank[k].tobytes() for k in bank)
    assert not (parent/'forced/turn_02/answer.md').exists()
    config=dict(old['runner_config']);config['resources']=dict(config['resources']);config['resume_root']=str(parent)
    parent_files=[p for p in parent.rglob('*') if p.is_file()]
    for p in parent_files+[ROOT/'docs/wspl/ATTENTION_PROBE_RESUME_V1.md']:config['resources'][str(p)]=sha(p)
    folder=ROOT/'data/attention_probe_resume_v1';folder.mkdir(exist_ok=False)
    target=json.loads(Path(config['target']).read_text());target['baseline_score']=json.loads((parent/'baseline_target.json').read_text())
    text=author_prompt(Path(config['guide']).read_text(),target,'forced',history,2)
    prompt=folder/'resume.prompt.txt';prompt.write_text(text,encoding='utf-8');config['resources'][str(prompt)]=sha(prompt)
    pf=folder/'preflight';pf.mkdir();(pf/'empty').mkdir()
    cli=json.loads(Path(config['local_paths']).read_text())['codex_cli'];preflight(cli,config['catalog'],config['instructions'],pf,text)
    for p,h in config['resources'].items():
        if sha(p)!=h:raise ValueError('resource changed: '+p)
    spec={k:v for k,v in old.items() if k not in ('files','runner_config')}
    spec.update(model_id='attention-probe-resume-v1',runner_config=config)
    rel='experiments/source_pins/attention_probe_resume_v1'
    for suffix in ('.spec.json','.manifest.json'):
        if (ROOT/(rel+suffix)).exists():raise ValueError('refuse overwrite')
    write_json(ROOT/(rel+'.spec.json'),spec)
    manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    isolated=ROOT.parent/'Skillforge-attention-probe-resume-v1';isolated.mkdir(exist_ok=False)
    for name in ('src','experiments','tests','tools'):shutil.copytree(ROOT/name,isolated/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',isolated/'AGENTS.md')
    h=sha(ROOT/(rel+'.manifest.json'));verify(isolated,isolated/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(config['resources']),parent_files=len(parent_files),
                completed_calls_preserved=2,interrupted_calls=1,interrupted_usage='unknown',max_new_calls=10,isolated=str(isolated),new_model_calls=0,new_predictions=0)
    write_json(ROOT/'docs/wspl/attention_probe_resume_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
