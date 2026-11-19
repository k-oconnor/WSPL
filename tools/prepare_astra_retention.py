"""Offline retention preparation. Never calls Astra or scores learned policies."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from run_closed_loop import file_hash
from skillforge.codex_one_shot import preflight
from skillforge.executive_study import load_library
from skillforge.wspl_artifact import ReLUArtifact
from skillforge.retention_pilot import DEFAULT_SOURCE,compile_selector,diagnose,packet
from skillforge.adaptation_pilot import episode,tasks
from skillforge.closed_loop_control import teacher
from skillforge.reliable_json import save


def main():
    parent=ROOT.parent/'WSPL-astra-adaptation-v1/results/astra_adaptation_v1'
    manifest=parent/'approved_manifest.json'
    assert file_hash(manifest)=='a01246c3f34579ad1c0dcac2dad890982bd8c67675a67c490ad956388a2aeb80'
    old=json.loads(manifest.read_text());c=old['runner_config'];resources={}
    for p,h in c['resources'].items():
        if file_hash(p)!=h:raise ValueError('Changed parent resource '+p)
        resources[p]=h
    repair=ReLUArtifact.load(parent/'movement_final.weights.json')
    frozen=json.loads((parent/'frozen.json').read_text());assert repair.sha256==frozen['movement']
    blocks,specs=load_library(c['library'])
    # Known synthetic engineering selector only; never sent to the author.
    selector=compile_selector(DEFAULT_SOURCE.replace('original = 1','original = dx').replace('repaired = 0','repaired = dy'))
    checks=[]
    for n in (3,5,8,12):
        task=tasks(500+n,[n])[0]
        for mapping,expected in [([[1,0],[0,1]],'original'),(c['mapping'],'repaired'),([[1,0],[0,1]],'original')]:
            e=episode(task,blocks,specs,mapping);selected,d=diagnose(e.world,[.005,0.],selector);assert selected==expected
            e.ticks+=1;e.blocks=dict(blocks,move=repair if selected=='repaired' else blocks['move'])
            while not e.finished:e.execute(teacher(e.observation()))
            assert e.result(False)['task_success'];checks.append(dict(objects=n,selected=selected,ticks=e.ticks))
    data=ROOT/'data/astra_retention_v1';data.mkdir(parents=True,exist_ok=True)
    catalog=data/'catalog_frozen.json';catalog.write_bytes(Path(c['catalog']).read_bytes())
    instructions=data/'instructions.md';instructions.write_text("Answer the user's request.\n",encoding='utf8')
    preview=data/('preflight_'+str(time.time_ns()));preview.mkdir();(preview/'empty').mkdir()
    preflight(c['cli'],catalog,instructions,preview,packet(blocks['move'],repair,[],dict(responses=8,evaluations=4)))
    for p in [manifest,parent/'movement_final.weights.json',parent/'frozen.json',catalog,instructions,Path(sys.executable),ROOT/'docs/wspl/ASTRA_RETENTION_V1.md']:
        resources[str(p.resolve())]=file_hash(p)
    config=dict(resources=resources,library=c['library'],mapping=c['mapping'],executives=c['executives'],
                executive_manifest=c['executive_manifest'],repair=str(parent/'movement_final.weights.json'),
                cli=c['cli'],catalog=str(catalog),instructions=str(instructions),wall_guard_seconds=8*3600)
    spec=dict(model_id='astra-retention-v1',source_roots=old['source_roots'],constructor=old['constructor'],
              constructor_kwargs=old['constructor_kwargs'],runner=dict(module='run_astra_retention',symbol='run',source='experiments/run_astra_retention.py'),runner_config=config)
    save(ROOT/'experiments/source_pins/astra_retention_v1.spec.json',spec)
    report=dict(status='prepared',resources=len(resources),engineering_scenes=checks,offline_prompt_passed=True,model_calls=0,approved=False)
    save(ROOT/'docs/wspl/astra_retention_preparation.json',report);print(json.dumps(dict(prepared=True,resources=len(resources),checks=len(checks))))


if __name__=='__main__':main()
