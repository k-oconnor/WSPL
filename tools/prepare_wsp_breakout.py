"""Freeze public prompts and a new unapproved spec. No model/game calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.wsp_breakout import ARMS,prompt
from skillforge.csail_breakout import dependency_identity,file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/wsp_breakout_v1')
    args=parser.parse_args();target=args.output.resolve();target.relative_to(ROOT)
    target.mkdir(parents=True,exist_ok=False)
    def relative(p):return p.relative_to(ROOT).as_posix()
    reference=ROOT/'data/csail_breakout_reference_v1/exp5_write_from_scratch/runs/breakout/message.txt'
    guide=ROOT/'docs/wspl/BREAKOUT_REWORKED_INTERFACE_V1.md'
    prompts={};records={}
    for arm in ARMS:
        p=target/(arm+'.txt')
        p.write_text(prompt(reference.read_text(encoding='utf-8'),guide.read_text(encoding='utf-8'),arm),encoding='utf-8',newline='\n')
        prompts[arm]=relative(p);records[relative(p)]=file_hash(p)
    bundle=target/'bundle.json'
    bundle.write_text(json.dumps(dict(prompts=prompts,prompt_hashes=records,
        guide_sha256=file_hash(guide),reference_prompt_sha256=file_hash(reference),
        model_calls=0,scored_episodes=0,reference_answers_supplied=False),indent=2)+'\n',encoding='utf-8')
    bound=[reference,guide,ROOT/'docs/wspl/BREAKOUT_REWORKED_V1.md',bundle,
        ROOT/'artifacts/local_paths.json',ROOT/'data/astra_native_v1/catalog_frozen.json',
        ROOT/'data/astra_native_v1/instructions.md',ROOT/'requirements/csail-breakout-replay.txt']
    records.update({relative(p):file_hash(p) for p in bound})
    cli=Path(json.loads((ROOT/'artifacts/local_paths.json').read_text(encoding='utf-8'))['codex_cli'])
    spec=dict(schema=1,model_id='wsp-breakout-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.wsp_breakout',source='src/skillforge/wsp_breakout.py',symbol='ReworkedBreakoutPolicy'),
        constructor_kwargs={},runner=dict(module='run_wsp_breakout',source='experiments/run_wsp_breakout.py',symbol='run'),
        runner_config=dict(bound_files=records,prompts=prompts,guide=relative(guide),reference_prompt=relative(reference),
            local_paths='artifacts/local_paths.json',catalog='data/astra_native_v1/catalog_frozen.json',
            instructions='data/astra_native_v1/instructions.md',cli_sha256=file_hash(cli),
            dependencies_sha256=dependency_identity()['sha256'],episodes=1000,max_steps=10000,
            call_guard_seconds=1200,wall_guard_seconds=14400))
    path=ROOT/'experiments/source_pins/wsp_breakout_v1.spec.json'
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(dict(spec=relative(path),planned_invocations=6,model_calls=0,scored_episodes=0,registered=False)))


if __name__=='__main__':main()
