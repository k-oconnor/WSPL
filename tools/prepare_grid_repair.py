"""Prepare pinned native adapter repair comparison; no model calls or scores."""
import json,sys,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,verify,write_json
from skillforge.grid_repair import prepare_data
from skillforge.grid_movement import encode_task
from skillforge.joint_attention import prompt
from skillforge.codex_one_shot import preflight
from skillforge.diffusion_edit import linux_path
from skillforge.go_replay import sha
from transformers import AutoTokenizer


def main():
    check=json.loads((ROOT/'docs/wspl/grid_repair_preflight.json').read_text());assert check['status']=='passed'
    old=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text());resources=dict(old['runner_config']['resources'])
    previous=json.loads((ROOT/'data/grid_movement_v1/domain.json').read_text());copy=json.loads((ROOT/'data/grid_copy_v1/copy.json').read_text())
    acquired,reserved=prepare_data(previous,copy);folder=ROOT/'data/grid_repair_v1';folder.mkdir(exist_ok=False)
    write_json(folder/'acquisition.json',acquired);write_json(folder/'reserved.json',reserved);write_json(folder/'demonstrations.json',previous['demonstrations'])
    kw=check['constructor_kwargs'];tok=AutoTokenizer.from_pretrained(kw['model_path'],local_files_only=True,trust_remote_code=False)
    lengths=[len(encode_task(tok,r,r['task'],previous['demonstrations'])[0]) for r in acquired+[r for rows in reserved.values() for r in rows]]
    guide=ROOT/'docs/wspl/GRID_REPAIR_INTERFACE_V1.md';protocol=ROOT/'docs/wspl/GRID_REPAIR_V1.md'
    local=ROOT/'artifacts/local_paths.json';cli=Path(json.loads(local.read_text())['codex_cli']);catalog=ROOT/'data/astra_native_v1/catalog_frozen.json';instructions=ROOT/'data/astra_native_v1/instructions.md';identity=ROOT/'data/go_authoring_v1/linux_runtime.json'
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(ROOT/'tools/construction_runtime_identity.py'),'--verify',linux_path(identity)],check=True,capture_output=True,timeout=90)
    text=prompt(guide.read_text(),dict(examples=acquired,note='Runtime adds acquisition-only current outputs and tensor shapes.'),[],0)
    (folder/'author.prompt.txt').write_text(text);pf=folder/'preflight';pf.mkdir();(pf/'empty').mkdir();preflight(cli,catalog,instructions,pf,text)
    parent=ROOT.parent/'Skillforge-grid-movement-v1/results/grid_movement_v1/last.pt'
    assert sha(parent)=='aef3a749ecb692f91e93aa789e95047735f7464858b337a3110983d37869a026'
    for path in [guide,protocol,local,cli,catalog,instructions,identity,parent,Path(kw['checkpoint_path']),folder/'acquisition.json',folder/'reserved.json',folder/'demonstrations.json',folder/'author.prompt.txt',ROOT/'docs/wspl/grid_repair_preflight.json']:resources[str(path)]=sha(path)
    for p,h in resources.items():
        if sha(p)!=h:raise ValueError('resource differs '+p)
    spec=dict(schema=1,model_id='grid-repair-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.grid_movement',source='src/skillforge/grid_movement.py',symbol='CopyFoundation'),constructor_kwargs=kw,
        runner=dict(module='run_grid_repair',source='experiments/run_grid_repair.py',symbol='run'),
        runner_config=dict(resources=resources,parent_adapter=str(parent),acquisition=str(folder/'acquisition.json'),reserved=str(folder/'reserved.json'),demonstrations=str(folder/'demonstrations.json'),
            guide=str(guide),local_paths=str(local),catalog=str(catalog),instructions=str(instructions),python_worker='tools/arc_python_worker.py',python_runtime='/home/kevin/venvs/skillforge-author',
            runtime_verify='tools/construction_runtime_identity.py',linux_identity=str(identity),wall_guard_seconds=14400))
    rel='experiments/source_pins/grid_repair_v1'
    for suffix in ('.spec.json','.manifest.json'):assert not (ROOT/(rel+suffix)).exists()
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    iso=ROOT.parent/'Skillforge-grid-repair-v1';iso.mkdir(exist_ok=False)
    for name in spec['source_roots']:shutil.copytree(ROOT/name,iso/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',iso/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(iso,iso/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),linux_resources=len(json.loads(identity.read_text())),isolated=str(iso),max_sequence_tokens=max(lengths),acquisition=len(acquired),reserved={k:len(v) for k,v in reserved.items()},author_calls=0,benchmark_predictions=0)
    write_json(ROOT/'docs/wspl/grid_repair_v1_preparation.json',report);print(json.dumps(report))

if __name__=='__main__':main()
