"""Freeze authoring prompts/resources and an UNAPPROVED source pin. No model calls."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.diffusion_edit import ARMS,linux_path,MODEL_IDS
from skillforge.diffusion_replay import sha
from skillforge.codex_one_shot import preflight
from source_consistency import inventory,write_json


def main():
    target=ROOT/'data/diffusion_edit_v1_ready';target.mkdir(exist_ok=False)
    old=json.loads((ROOT/'experiments/source_pins/diffusion_replay_v1.spec.json').read_text())
    resources={k:v for k,v in old['runner_config']['resources'].items() if Path(k).is_absolute()}
    runtime_tool=ROOT/'tools/construction_runtime_identity.py'
    identity=target/'linux_runtime.json'
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(runtime_tool),'--write',linux_path(identity)],check=True)
    guide=ROOT/'docs/wspl/DIFFUSION_EDIT_INTERFACE_V1.md'
    local=ROOT/'artifacts/local_paths.json';cli=Path(json.loads(local.read_text())['codex_cli'])
    catalog=ROOT/'data/astra_native_v1/catalog_frozen.json';instructions=ROOT/'data/astra_native_v1/instructions.md'
    paths=[guide,ROOT/'docs/wspl/DIFFUSION_EDIT_V1.md',identity,local,cli,catalog,instructions]
    prompts={}
    for arm in ARMS:
        text=guide.read_text(encoding='utf-8')+'\n\nSELECTED ARM: '+arm+'\n'
        path=target/(arm+'.txt');path.write_text(text,encoding='utf-8');prompts[arm]=str(path)
        check=target/(arm+'_preflight');check.mkdir();(check/'empty').mkdir();preflight(cli,catalog,instructions,check,text)
        paths.append(path)
    resources.update({str(p):sha(p) for p in paths})
    spec=dict(schema=1,model_id='diffusion-edit-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.diffusion_edit',source='src/skillforge/diffusion_edit.py',symbol='DiffusionEditStudy'),
        constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_diffusion_edit',source='experiments/run_diffusion_edit.py',symbol='run'),
        runner_config=dict(resources=resources,prompts=prompts,worker='tools/python_construction_worker.py',runtime_tool='tools/construction_runtime_identity.py',
            linux_runtime='/home/kevin/venvs/skillforge-author',linux_identity=str(identity),local_paths=str(local),catalog=str(catalog),instructions=str(instructions),
            call_guard_seconds=1200,wall_guard_seconds=14400))
    path=ROOT/'experiments/source_pins/diffusion_edit_v1.spec.json'
    manifest=path.with_name('diffusion_edit_v1.manifest.json')
    if path.exists() or manifest.exists():raise ValueError('refuse overwrite')
    write_json(path,spec);files=inventory(ROOT,spec['source_roots']);write_json(manifest,{**spec,'files':files})
    print(json.dumps(dict(status='UNAPPROVED',sha256=sha(manifest),sources=len(files),windows_resources=len(resources),linux_resources=len(json.loads(identity.read_text())),model_calls=0)))


if __name__=='__main__': main()
