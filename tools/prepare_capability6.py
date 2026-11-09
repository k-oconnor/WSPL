"""Freeze capability6 data/source/runtime identity; no scientific execution."""
import sys,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,verify,write_json
from skillforge.capability_data import prepare
from skillforge.capability_codec import encode
from skillforge.capability_bridge import default_plan,validate_plan
from skillforge.codex_one_shot import preflight
from skillforge.diffusion_edit import linux_path
from skillforge.go_replay import sha
from run_capability6 import author_prompt
from transformers import AutoTokenizer


def main():
    check=json.loads((ROOT/'docs/wspl/capability6_preflight.json').read_text());assert check['status']=='passed'
    old=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text());resources=dict(old['runner_config']['resources']);kw=check['constructor_kwargs']
    copy_path=ROOT/'data/grid_copy_v1/copy.json';copy=json.loads(copy_path.read_text());excluded=[r['input'] for rows in copy.values() for r in rows]
    assert not (ROOT/'experiments/source_pins/capability6_v1.manifest.json').exists()
    data=prepare(excluded=excluded);folder=ROOT/'data/capability6_v1';folder.mkdir(exist_ok=True)
    for name in ('train','dev','test','shape_families'):write_json(folder/(name+'.json'),data[name])
    tok=AutoTokenizer.from_pretrained(kw['model_path'],local_files_only=True,trust_remote_code=False)
    longest=max(len(encode(tok,r)[0]) for split in ('train','dev','test') for rows in data[split].values() for r in rows)
    plan=default_plan();metadata=validate_plan(plan)
    write_json(folder/'default_metadata.json',metadata)
    base_guide=(ROOT/'docs/wspl/CAPABILITY6_INTERFACE_V1.md').read_text();graph_guide=(ROOT/'docs/wspl/WSP_GRAPH_SURGERY.md').read_text()
    local=ROOT/'artifacts/local_paths.json';cli=Path(json.loads(local.read_text())['codex_cli']);catalog=ROOT/'data/astra_native_v1/catalog_frozen.json';instructions=ROOT/'data/astra_native_v1/instructions.md';identity=ROOT/'data/go_authoring_v1/linux_runtime.json'
    # One guide for both interfaces; native availability is explicit and audited by interface.
    guide=folder/'guide.md';guide.write_text(base_guide+'\nWSPL GRAPH REFERENCE (WSPL arm only):\n'+graph_guide)
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(ROOT/'tools/construction_runtime_identity.py'),'--verify',linux_path(identity)],check=True,capture_output=True,timeout=90)
    packet=dict(training_counts={k:len(v) for k,v in data['train'].items()},development_counts={k:len(v) for k,v in data['dev'].items()},examples={k:v[:8] for k,v in data['train'].items()},development_examples={k:v[:8] for k,v in data['dev'].items()},default_parameters=metadata['parameters'],budget=dict(updates=0,train_seconds=0.,targets=0))
    for interface in ('native','wspl'):
        text=author_prompt(guide.read_text(),interface,packet,[],0);(folder/(interface+'.prompt.txt')).write_text(text)
        pf=folder/(interface+'_preflight');pf.mkdir();(pf/'empty').mkdir();preflight(cli,catalog,instructions,pf,text)
    bound=[ROOT/'docs/wspl/CAPABILITY6_V1.md',ROOT/'docs/wspl/CAPABILITY6_INTERFACE_V1.md',ROOT/'docs/wspl/capability6_preflight.json',copy_path,local,cli,catalog,instructions,identity,Path(kw['checkpoint_path'])]
    bound+=list(folder.glob('*.json'))+list(folder.glob('*.txt'))+[guide]
    for p in bound:resources[str(p)]=sha(p)
    for p,h in resources.items():
        if sha(p)!=h:raise ValueError('resource differs '+p)
    spec=dict(schema=1,model_id='capability6-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.grid_movement',source='src/skillforge/grid_movement.py',symbol='CopyFoundation'),constructor_kwargs=kw,
        runner=dict(module='run_capability6',source='experiments/run_capability6.py',symbol='run'),
        runner_config=dict(resources=resources,train=str(folder/'train.json'),dev=str(folder/'dev.json'),test=str(folder/'test.json'),guide=str(guide),local_paths=str(local),catalog=str(catalog),instructions=str(instructions),
          worker='tools/capability_worker.py',python_runtime='/home/kevin/venvs/skillforge-author',runtime_verify='tools/construction_runtime_identity.py',linux_identity=str(identity),wall_guard_seconds=64800))
    rel='experiments/source_pins/capability6_v1'
    for suffix in ('.spec.json','.manifest.json'):assert not (ROOT/(rel+suffix)).exists()
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    iso=ROOT.parent/'Skillforge-capability6-v1';iso.mkdir(exist_ok=False)
    for name in spec['source_roots']:shutil.copytree(ROOT/name,iso/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',iso/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(iso,iso/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),linux_resources=len(json.loads(identity.read_text())),isolated=str(iso),max_sequence_tokens=longest,
        default_parameters=metadata['parameters'],rows={split:{k:len(v) for k,v in data[split].items()} for split in ('train','dev','test')},author_calls=0,benchmark_predictions=0)
    write_json(ROOT/'docs/wspl/capability6_v1_preparation.json',report);print(json.dumps(report))

if __name__=='__main__':main()
