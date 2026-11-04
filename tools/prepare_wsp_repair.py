"""Freeze repair tasks/prompts and an unapproved specification. Never calls a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.wsp_repair import TASKS,ARMS,public_task,prompt
from skillforge.wsp_artifact import digest
from skillforge.breakout_matched import matched_dependency_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'data/wsp_repair_v1');args=parser.parse_args()
    target=args.output.resolve();target.relative_to(ROOT);target.mkdir(parents=True,exist_ok=False)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def relative(p):return p.relative_to(ROOT).as_posix()
    guide=ROOT/'docs/wspl/REPAIR_INTERFACE_V1.md';text=guide.read_text(encoding='utf-8');prompts={};records={}
    for task in TASKS:
        prompts[task]={}
        for arm in ARMS:
            p=target/f'{task}.{arm}.txt';p.write_text(prompt(task,arm,text),encoding='utf-8',newline='\n');prompts[task][arm]=relative(p);records[relative(p)]=sha(p)
    task_ids={t:digest(public_task(t)) for t in TASKS}
    bundle=target/'bundle.json';bundle.write_text(json.dumps(dict(task_ids=task_ids,prompts=prompts,prompt_hashes=records,guide_sha256=sha(guide),model_calls=0),indent=2)+'\n',encoding='utf-8')
    bound=[ROOT/'artifacts/local_paths.json',ROOT/'data/astra_native_v1/catalog_frozen.json',ROOT/'data/astra_native_v1/instructions.md',guide,ROOT/'docs/wspl/REPAIR_SCREEN_V1.md',ROOT/'requirements/breakout-matched.txt',ROOT/'requirements/csail-breakout-replay.txt',bundle]
    records.update({relative(p):sha(p) for p in bound})
    cli=Path(json.loads((ROOT/'artifacts/local_paths.json').read_text())['codex_cli'])
    spec=dict(schema=1,model_id='wsp-repair-v1',source_roots=['src','experiments','tests','tools'],
              constructor=dict(module='skillforge.wsp_repair',source='src/skillforge/wsp_repair.py',symbol='RepairStudy'),constructor_kwargs={},
              runner=dict(module='run_wsp_repair',source='experiments/run_wsp_repair.py',symbol='run'),
              runner_config=dict(bound_files=records,prompts=prompts,guide=relative(guide),task_ids=task_ids,
                local_paths='artifacts/local_paths.json',catalog='data/astra_native_v1/catalog_frozen.json',instructions='data/astra_native_v1/instructions.md',
                cli_sha256=sha(cli),dependencies_sha256=matched_dependency_identity()['sha256'],call_guard_seconds=1200,wall_guard_seconds=14400))
    path=ROOT/'experiments/source_pins/wsp_repair_v1.spec.json'
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(dict(spec=relative(path),prompts=6,planned_invocations=12,model_calls=0,registered=False)))


if __name__=='__main__':main()
