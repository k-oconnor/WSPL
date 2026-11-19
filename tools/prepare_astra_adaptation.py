"""Offline source/resource preparation; no author calls or scientific predictions."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from run_closed_loop import file_hash
from skillforge.codex_one_shot import preflight
from skillforge.executive_study import load_library
from skillforge.controller_author import teacher,construct
from skillforge.closed_loop_control import teacher as reference_policy
from skillforge.adaptation_pilot import episode,tasks,packet
from skillforge.reliable_json import save


def main():
    parent=ROOT.parent/'WSPL-closed-loop-v1/results/compact_closed_loop_v1'
    manifest=parent/'approved_manifest.json'
    if file_hash(manifest)!='c5dacefd62c61ebcd9fcd439be9dca8e51ebda4620a65a7516f1aed6aefd90e3':raise ValueError('Parent identity')
    old=json.loads(manifest.read_text());c=old['runner_config'];resources={}
    for path,h in c['resources'].items():
        if file_hash(path)!=h:raise ValueError('Changed resource '+path)
        resources[path]=h
    blocks,specs=load_library(c['library'])
    mapping=[[0.,-.8],[1.2,0.]]
    refsource=teacher(specs['move']).replace('0.73 * x','0.6 * y').replace('0.73 * y','-0.9 * x')
    ref=construct(refsource,specs['move']);scenes=[]
    for n in [3,5,8,12]:
        t=tasks(400+n,[n])[0];e=episode(t,dict(blocks,move=ref),specs,mapping)
        while not e.finished:e.execute(reference_policy(e.observation()))
        assert e.result(False)['task_success'];scenes.append(dict(objects=n,ticks=e.ticks))
    previous=json.loads((ROOT.parent/'Skillforge-python-addition-v1/experiments/source_pins/python_addition_v1.spec.json').read_text())['runner_config']
    data=ROOT/'data/astra_adaptation_v1';data.mkdir(parents=True,exist_ok=True)
    catalog=data/'catalog_frozen.json';catalog.write_bytes(Path(previous['catalog']).read_bytes())
    instructions=data/'instructions.md';instructions.write_text("Answer the user's request.\n",encoding='utf8')
    cli=Path('C:/Users/kevin/.vscode/extensions/openai.chatgpt-26.1111.21308-win32-x64/bin/windows-x86_64/codex.exe')
    executives={str(s):str(parent/f'seed_{s}/adapter_final.pt') for s in [113,227,341]}
    import torch
    for p in executives.values():
        saved=torch.load(p,map_location='cpu',weights_only=False)
        assert saved['manifest_sha256']=='c5dacefd62c61ebcd9fcd439be9dca8e51ebda4620a65a7516f1aed6aefd90e3'
        assert saved['base_hash']==json.loads((parent/'results.json').read_text())['base_hash']
        del saved
    for p in [manifest,*map(Path,executives.values()),catalog,instructions,cli,Path(sys.executable),ROOT/'docs/wspl/ASTRA_ADAPTATION_V1.md']:
        resources[str(p.resolve())]=file_hash(p)
    # Bind the native CLI directory, not just its entry point.
    for p in cli.parent.rglob('*'):
        if p.is_file():resources[str(p.resolve())]=file_hash(p)
    import time
    preview=data/('preflight_'+str(time.time_ns()));preview.mkdir();(preview/'empty').mkdir()
    preflight(cli,catalog,instructions,preview,packet(blocks['move'],specs,[],dict(turns=12,probe_commands=32,author_calls=4,fit_updates=4096,evaluations=4)))
    config=dict(resources=resources,library=c['library'],mapping=mapping,reference_source=refsource,
                executives=executives,executive_manifest='c5dacefd62c61ebcd9fcd439be9dca8e51ebda4620a65a7516f1aed6aefd90e3',
                author_adapter=c['author_adapter'],author_manifest='5de7ada4ce5d2c3c41614896e568cca4b91bc63991f5f4ce997a7e9404d1f3ae',
                cli=str(cli),catalog=str(catalog),instructions=str(instructions),wall_guard_seconds=8*3600)
    spec=dict(model_id='astra-adaptation-v1',source_roots=old['source_roots'],constructor=old['constructor'],
              constructor_kwargs=old['constructor_kwargs'],runner=dict(module='run_astra_adaptation',symbol='run',source='experiments/run_astra_adaptation.py'),runner_config=config)
    save(ROOT/'experiments/source_pins/astra_adaptation_v1.spec.json',spec)
    save(ROOT/'docs/wspl/astra_adaptation_preparation.json',dict(status='prepared',resources=len(resources),engineering_scenes=scenes,
         offline_prompt_passed=True,scientific_runs=0,model_calls=0,approved=False))
    print(json.dumps(dict(prepared=True,resources=len(resources),approved=False)))


if __name__=='__main__':main()
