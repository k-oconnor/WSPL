"""Prepare the frozen grid interface gate; no domain scoring or training."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,verify,write_json
from skillforge.grid_domain import prepare_sets
from skillforge.grid_io import render_prompt,palette_ids
from skillforge.go_replay import sha
from transformers import AutoTokenizer


def main():
    check=json.loads((ROOT/'docs/wspl/grid_io_preflight.json').read_text());assert check['status']=='passed'
    old=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text());resources=dict(old['runner_config']['resources'])
    data=prepare_sets();folder=ROOT/'data/grid_io_v1';folder.mkdir(exist_ok=False);write_json(folder/'domain.json',data)
    tok=AutoTokenizer.from_pretrained(old['constructor_kwargs']['model_path'],local_files_only=True,trust_remote_code=False);palette=palette_ids(tok)
    lengths=[len(tok.encode(render_prompt(row['input'],'copy' if name=='copy' else 'movement',data['demonstrations']),add_special_tokens=False)) for name in ('copy','movement_familiar','movement_novel') for row in data[name]]
    assert max(lengths)<=768
    for path in [folder/'domain.json',ROOT/'docs/wspl/GRID_DOMAIN_ADAPTER_V1.md',ROOT/'docs/wspl/grid_io_preflight.json']:resources[str(path)]=sha(path)
    for path,h in resources.items():
        if sha(path)!=h:raise ValueError('resource differs '+path)
    spec=dict(schema=1,model_id='grid-io-v1',source_roots=['src','experiments','tests','tools'],constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
              runner=dict(module='run_grid_io',source='experiments/run_grid_io.py',symbol='run'),
              runner_config=dict(resources=resources,data=str(folder/'domain.json'),model_sha256=check['model_sha256'],wall_guard_seconds=7200))
    rel='experiments/source_pins/grid_io_v1'
    for suffix in ('.spec.json','.manifest.json'):assert not (ROOT/(rel+suffix)).exists()
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    iso=ROOT.parent/'Skillforge-grid-io-v1';iso.mkdir(exist_ok=False)
    for name in spec['source_roots']:shutil.copytree(ROOT/name,iso/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',iso/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(iso,iso/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),isolated=str(iso),rows={k:len(v) for k,v in data.items()},palette_ids=palette,max_prefix_tokens=max(lengths),domain_predictions=0,author_calls=0)
    write_json(ROOT/'docs/wspl/grid_io_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
