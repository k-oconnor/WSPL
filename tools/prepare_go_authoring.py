"""Create unapproved matched Go authoring prompts/spec; never author or evaluate."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.go_authoring import conversation_prompt
from skillforge.codex_one_shot import preflight
from skillforge.go_replay import sha
from source_consistency import inventory,write_json


def main():
    target=ROOT/'data/go_authoring_v1';target.mkdir(exist_ok=False)
    go=json.loads((ROOT/'experiments/source_pins/go_replay_v1.spec.json').read_text())
    diffusion=json.loads((ROOT/'experiments/source_pins/diffusion_edit_v1_final.spec.json').read_text())
    linux={p:h for p,h in go['runner_config']['resources'].items() if p.startswith('/')}
    linux.update(json.loads(Path(diffusion['runner_config']['linux_identity']).read_text()))
    linux_identity=target/'linux_runtime.json';write_json(linux_identity,linux)
    old=json.loads((ROOT/'experiments/source_pins/diffusion_replay_v1.spec.json').read_text())
    windows={p:h for p,h in old['runner_config']['resources'].items() if ('WindowsApps' in p or 'venv-sim' in p)}
    local=ROOT/'artifacts/local_paths.json';cli=Path(json.loads(local.read_text())['codex_cli'])
    catalog=ROOT/'data/astra_native_v1/catalog_frozen.json';instructions=ROOT/'data/astra_native_v1/instructions.md'
    guide=ROOT/'docs/wspl/GO_AUTHORING_INTERFACE_V1.md'
    bound=[local,cli,catalog,instructions,guide,ROOT/'docs/wspl/GO_AUTHORING_V1.md',linux_identity]
    for arm in ('python','wspl'):
        text=conversation_prompt(guide.read_text(encoding='utf-8'),arm,[],0)
        path=target/(arm+'.txt');path.write_text(text,encoding='utf-8');bound.append(path)
        folder=target/(arm+'_preflight');folder.mkdir();(folder/'empty').mkdir();preflight(cli,catalog,instructions,folder,text)
    windows.update({str(p):sha(p) for p in bound})
    spec=dict(schema=1,model_id='go-authoring-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.go_authoring',source='src/skillforge/go_authoring.py',symbol='GoAuthoringStudy'),constructor_kwargs={},
        runner=dict(module='run_go_authoring',source='experiments/run_go_authoring.py',symbol='run'),
        runner_config=dict(windows_resources=windows,linux_resources=linux,linux_identity=str(linux_identity),
            guide=str(guide),local_paths=str(local),catalog=str(catalog),instructions=str(instructions),
            python_worker='tools/go_python_worker.py',wspl_worker='tools/go_wspl_worker.py',eval_worker='tools/evaluate_go_candidate.py',
            runtime_verify='tools/construction_runtime_identity.py',python_runtime='/home/kevin/venvs/skillforge-author',
            go_python='/home/kevin/venvs/skillforge-go/bin/python',opponent_root=go['constructor_kwargs']['reference_root'],
            wall_guard_seconds=21600))
    path=ROOT/'experiments/source_pins/go_authoring_v1.spec.json';manifest=path.with_name('go_authoring_v1.manifest.json')
    if path.exists() or manifest.exists():raise ValueError('refuse overwrite')
    write_json(path,spec);files=inventory(ROOT,spec['source_roots']);write_json(manifest,{**spec,'files':files})
    print(json.dumps(dict(status='UNAPPROVED',sha256=sha(manifest),sources=len(files),windows_resources=len(windows),linux_resources=len(linux),model_calls=0,games=0)))


if __name__=='__main__':main()
