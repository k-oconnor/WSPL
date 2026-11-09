"""Prepare frozen ARC transfer identity; no scored predictions or launch."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,write_json,verify
from skillforge.go_replay import sha
from skillforge.arc_eval import encode_choices
from run_capability_arc import load_rows
from transformers import AutoTokenizer


def main():
    check=json.loads((ROOT/'docs/wspl/capability_arc_preflight.json').read_text())
    assert check['status']=='passed' and check['benchmark_predictions']==0
    parent=ROOT.parent/'Skillforge-capability6-v2/results/capability6_v1'
    assert json.loads((parent/'results.json').read_text())['status']=='complete'
    old=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text())
    # Reuse the previously inventoried model/runtime identity, not authoring services.
    resources={p:h for p,h in old['runner_config']['resources'].items() if not Path(p).is_relative_to(ROOT)}
    foundation=check['foundation'];frozen=json.loads((parent/'frozen.json').read_text())
    adapters={arm:str(parent/arm/'final.json') for arm in frozen}
    for arm,path in adapters.items():
        assert sha(path)==frozen[arm]==check['arms'][arm]['final_sha256']
    folder=Path('C:/Users/kevin/skillforge-runtime/datasets/arc_authoring_v1')
    datasets={split:dict(path=str(folder/(split+'.jsonl')),n=n) for split,n in [('challenge_test',1172),('easy_test',2376)]}
    tokenizer=AutoTokenizer.from_pretrained(foundation['model_path'],local_files_only=True,trust_remote_code=False)
    longest=0
    for data in datasets.values():
        for row in load_rows(data['path'],data['n']):
            longest=max(longest,max(len(ids) for ids,_ in encode_choices(tokenizer,row)))
    bound=[ROOT/'docs/wspl/CAPABILITY_ARC_V1.md',ROOT/'docs/wspl/capability_arc_preflight.json',
        Path(foundation['checkpoint_path']),parent/'frozen.json',parent/'results.json']
    bound+=list(map(Path,adapters.values()))+[Path(d['path']) for d in datasets.values()]
    for path in bound:resources[str(path)]=sha(path)
    for path,h in resources.items():
        if sha(path)!=h:raise ValueError('resource changed: '+path)
    spec=dict(schema=1,model_id='capability-arc-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.arc_eval',source='src/skillforge/arc_eval.py',symbol='ArcBaseline'),
        constructor_kwargs=dict(model_path=foundation['model_path']),
        runner=dict(module='run_capability_arc',source='experiments/run_capability_arc.py',symbol='run'),
        runner_config=dict(resources=resources,foundation=foundation,original_model_sha256=check['original_model_sha256'],
            adapters=adapters,datasets=datasets,wall_guard_seconds=14400))
    rel='experiments/source_pins/capability_arc_v1'
    for suffix in ('.spec.json','.manifest.json'):assert not (ROOT/(rel+suffix)).exists()
    write_json(ROOT/(rel+'.spec.json'),spec)
    manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    iso=ROOT.parent/'Skillforge-capability-arc-v1';iso.mkdir(exist_ok=False)
    for name in spec['source_roots']:shutil.copytree(ROOT/name,iso/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',iso/'AGENTS.md')
    h=sha(ROOT/(rel+'.manifest.json'));verify(iso,iso/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),isolated=str(iso),
        row_condition_evaluations=7*sum(d['n'] for d in datasets.values()),max_sequence_tokens=longest,benchmark_predictions=0,author_calls=0)
    write_json(ROOT/'docs/wspl/capability_arc_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
