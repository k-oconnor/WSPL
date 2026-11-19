"""Bind teacher/student resources and offline prompt; no scientific execution."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from run_executive_teaching import sha
from skillforge.executive_teaching import teacher_prompt
from skillforge.student_control import prompt
from skillforge.codex_one_shot import preflight
from skillforge.reliable_json import save


def main():
    factory=ROOT.parent/'WSPL-factory-v1/results/factory_pilot_v1'
    retention=ROOT.parent/'WSPL-astra-retention-v1/results/astra_retention_v1'
    expected=[(factory/'approved_manifest.json','3af5323cd31a9b7babc5bd012c1ce9896e9010f5acae4473fd8f5bdae5a2af48'),
              (retention/'approved_manifest.json','4ca2229cd1bcdd9852accad248b34eb34edf335f1436d6b2a98aa9d109874eb4')]
    resources={};parents=[]
    for path,h in expected:
        if sha(path)!=h:raise ValueError('Historical manifest changed')
        m=json.loads(path.read_text());parents.append(m)
        for p,value in m['runner_config']['resources'].items():
            if sha(p)!=value:raise ValueError('Historical resource changed '+p)
            if p in resources and resources[p]!=value:raise ValueError('Conflicting runtime identity')
            resources[p]=value
    c=parents[1]['runner_config']
    motor=factory/'artifact_final.json'
    if sha(motor)!=json.loads((factory/'frozen.json').read_text())['artifact_sha256']:raise ValueError('Motor artifact changed')
    parent=ROOT.parent/'WSPL-closed-loop-v1/results/compact_closed_loop_v1/seed_113/adapter_final.pt'
    data=ROOT/'data/executive_teaching_v1';data.mkdir(parents=True,exist_ok=True)
    catalog=data/'catalog_frozen.json';catalog.write_bytes(Path(c['catalog']).read_bytes())
    instructions=data/'instructions.md';instructions.write_text("Answer the user's request.\n",encoding='utf8')
    public=json.loads((ROOT/'data/teaching_preflight/public_fixture.json').read_text())
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('C:/Users/kevin/skillforge-runtime/models/Qwen2.5-Coder-1.5B-Instruct',local_files_only=True)
    student=prompt(public['observation'],public['documentation'])
    rendered=tokenizer.apply_chat_template([dict(role='user',content=student)],tokenize=False,add_generation_prompt=True)
    ids=tokenizer(rendered,add_special_tokens=False)['input_ids']
    if len(ids)+96>4096:raise ValueError('Engineering prompt too long')
    preview=data/('preflight_'+str(time.time_ns()));preview.mkdir();(preview/'empty').mkdir()
    preflight(c['cli'],catalog,instructions,preview,teacher_prompt(public))
    for p in [parent,motor,factory/'frozen.json',catalog,instructions,Path(sys.executable),ROOT/'docs/wspl/EXECUTIVE_TEACHING_V1.md']:
        resources[str(p.resolve())]=sha(p)
    config=dict(resources=resources,parent=str(parent),parent_manifest='c5dacefd62c61ebcd9fcd439be9dca8e51ebda4620a65a7516f1aed6aefd90e3',
        motor_artifact=str(motor),sim_python='C:/Users/kevin/skillforge-runtime/venv-sim/Scripts/python.exe',
        sim_worker=str(ROOT/'tools/teaching_sim_server.py'),cli=c['cli'],catalog=str(catalog),instructions=str(instructions),wall_guard_seconds=43200)
    spec=dict(model_id='executive-teaching-v1',source_roots=['src','experiments','tools','tests'],
        constructor=parents[1]['constructor'],constructor_kwargs=parents[1]['constructor_kwargs'],
        runner=dict(module='run_executive_teaching',symbol='run',source='experiments/run_executive_teaching.py'),runner_config=config)
    save(ROOT/'experiments/source_pins/executive_teaching_v1.spec.json',spec)
    checks={name:json.loads((ROOT/f'docs/wspl/teaching_{name}_preflight.json').read_text()) for name in ('physics','student','rpc')}
    assert all(v['status']=='passed' for v in checks.values())
    report=dict(status='prepared',approved=False,model_calls=0,resources=len(resources),engineering_prompt_tokens=len(ids),checks=checks)
    save(ROOT/'docs/wspl/executive_teaching_preparation.json',report)
    print(json.dumps(dict(prepared=True,resources=len(resources),engineering_prompt_tokens=len(ids),model_calls=0)))


if __name__=='__main__':main()
