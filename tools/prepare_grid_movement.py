"""Prepare isolated movement adapter experiment, without running it."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,verify,write_json
from skillforge.grid_movement import prepare_data,encode_task
from skillforge.go_replay import sha
from transformers import AutoTokenizer


def main():
    check=json.loads((ROOT/'docs/wspl/grid_movement_preflight.json').read_text());assert check['status']=='passed'
    old=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text());resources=dict(old['runner_config']['resources'])
    original=json.loads((ROOT/'data/grid_io_v1/domain.json').read_text());copy=json.loads((ROOT/'data/grid_copy_v1/copy.json').read_text())
    data=prepare_data(original,copy);folder=ROOT/'data/grid_movement_v1';folder.mkdir(exist_ok=False);write_json(folder/'domain.json',data)
    kw=check['constructor_kwargs'];tok=AutoTokenizer.from_pretrained(kw['model_path'],local_files_only=True,trust_remote_code=False)
    lengths=[len(encode_task(tok,r,'movement',data['demonstrations'])[0]) for r in data['train']+data['familiar']+data['novel']]
    for p in [folder/'domain.json',ROOT/'docs/wspl/GRID_MOVEMENT_V1.md',ROOT/'docs/wspl/grid_movement_preflight.json',Path(kw['checkpoint_path'])]:resources[str(p)]=sha(p)
    assert resources[kw['checkpoint_path']]==kw['checkpoint_sha256']
    for p,h in resources.items():
        if sha(p)!=h:raise ValueError('resource differs '+p)
    spec=dict(schema=1,model_id='grid-movement-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.grid_movement',source='src/skillforge/grid_movement.py',symbol='CopyFoundation'),constructor_kwargs=kw,
        runner=dict(module='run_grid_movement',source='experiments/run_grid_movement.py',symbol='run'),
        runner_config=dict(resources=resources,data=str(folder/'domain.json'),wall_guard_seconds=7200,steps=1024,lr=.001))
    rel='experiments/source_pins/grid_movement_v1'
    for suffix in ('.spec.json','.manifest.json'):assert not (ROOT/(rel+suffix)).exists()
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    iso=ROOT.parent/'Skillforge-grid-movement-v1';iso.mkdir(exist_ok=False)
    for name in spec['source_roots']:shutil.copytree(ROOT/name,iso/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',iso/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(iso,iso/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),isolated=str(iso),max_movement_sequence_tokens=max(lengths),rows={k:len(v) for k,v in data.items()},domain_predictions=0,training_runs=0,author_calls=0)
    write_json(ROOT/'docs/wspl/grid_movement_v1_preparation.json',report);print(json.dumps(report))

if __name__=='__main__':main()
