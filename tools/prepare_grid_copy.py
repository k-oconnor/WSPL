"""Prepare copy-only foundation pin; no training or benchmark inference."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,verify,write_json
from skillforge.grid_domain import enumerate_domain
from skillforge.grid_copy import make_data,encode_copy
from skillforge.go_replay import sha
from transformers import AutoTokenizer


def main():
    check=json.loads((ROOT/'docs/wspl/grid_copy_preflight.json').read_text());assert check['status']=='passed'
    old=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text());resources=dict(old['runner_config']['resources'])
    previous=json.loads((ROOT/'data/grid_io_v1/domain.json').read_text())
    data=make_data([r['input'] for r in enumerate_domain()]);data['demonstrations']=[dict(input=r['input']) for r in previous['demonstrations']]
    data['historical_copy']=[dict(input=r['input']) for r in previous['copy']]
    folder=ROOT/'data/grid_copy_v1';folder.mkdir(exist_ok=False);write_json(folder/'copy.json',data)
    tok=AutoTokenizer.from_pretrained(old['constructor_kwargs']['model_path'],local_files_only=True,trust_remote_code=False)
    lengths=[len(encode_copy(tok,r['input'],data['demonstrations'])[0]) for r in data['train']+data['heldout']]
    for p in [folder/'copy.json',ROOT/'docs/wspl/GRID_COPY_V1.md',ROOT/'docs/wspl/grid_copy_preflight.json']:resources[str(p)]=sha(p)
    for p,h in resources.items():
        if sha(p)!=h:raise ValueError('resource differs '+p)
    spec=dict(schema=1,model_id='grid-copy-v1',source_roots=['src','experiments','tests','tools'],constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_grid_copy',source='experiments/run_grid_copy.py',symbol='run'),
        runner_config=dict(resources=resources,data=str(folder/'copy.json'),model_sha256=check['model_sha256'],wall_guard_seconds=7200,steps=512,batch_size=4,lr=.00002))
    rel='experiments/source_pins/grid_copy_v1'
    for suffix in ('.spec.json','.manifest.json'):assert not (ROOT/(rel+suffix)).exists()
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    iso=ROOT.parent/'Skillforge-grid-copy-v1';iso.mkdir(exist_ok=False)
    for name in spec['source_roots']:shutil.copytree(ROOT/name,iso/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',iso/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(iso,iso/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),isolated=str(iso),max_sequence_tokens=max(lengths),domain_predictions=0,training_runs=0,author_calls=0)
    write_json(ROOT/'docs/wspl/grid_copy_v1_preparation.json',report);print(json.dumps(report))

if __name__=='__main__':main()
