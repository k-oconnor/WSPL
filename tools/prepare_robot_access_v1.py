"""Prepare concrete broader-access source pin; never approve/register/launch."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify,inventory,write_json
from skillforge.fresh_robot import sha


def main():
    target=ROOT/'experiments/source_pins/robot_authoring_access_v1.manifest.json'
    if target.exists():raise ValueError('Preserve existing candidate identity')
    oldroot=ROOT.parent/'WSPL-robot-authoring-v2'
    prior_manifest=oldroot/'experiments/source_pins/robot_authoring_v2.manifest.json'
    old=verify(oldroot,prior_manifest,'6640c6968490e742abbc619f3619207d9047d1171cd0f3328207e93fa68354bb')
    for name in ('robot_trials','robot_codec','robot_submissions','wspl_lora','authoring_sft','robot_collector','robot_recovery'):
        path='src/skillforge/'+name+'.py'
        if sha(ROOT/path)!=old['files'][path]:raise ValueError('Unexpected task/runtime change: '+path)
    resources=dict(old['runner_config']['resources'])
    reports=['robot_access_unit_checks.json','robot_access_model_preflight.json','robot_access_prompt_preflight.json']
    for name in reports:
        path=ROOT/'docs/wspl'/name
        if json.loads(path.read_text())['status']!='passed':raise ValueError('Missing passing check: '+name)
        resources[str(path)]=sha(path)
    for path in [prior_manifest,ROOT/'docs/wspl/ROBOT_AUTHORING_ACCESS_V1.md',
                 ROOT/'data/robot_authoring_access_v1/catalog.json',ROOT/'data/robot_authoring_access_v1/instructions.md']:
        resources[str(path)]=sha(path)
    def check(item):
        path,expected=item
        if sha(path)!=expected:raise ValueError('Changed bound resource: '+path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(check,resources.items()):pass
    cfg=dict(old['runner_config'])
    cfg.update(resources=resources,author_responses=24,inspection_attempts=8,
        inspection_elements=64_000_000,inspection_seconds=300,final_seed=73600000,
        development_wall_seconds=86400,prior_compute_seconds=0,prior_elapsed_seconds=0,
        sim_worker=str(ROOT/'tools/teaching_sim_server.py'),codec_source=str(ROOT/'src/skillforge/robot_codec.py'),
        submission_worker=str(ROOT/'tools/robot_submission_worker.py'),
        catalog=str(ROOT/'data/robot_authoring_access_v1/catalog.json'),instructions=str(ROOT/'data/robot_authoring_access_v1/instructions.md'))
    spec=dict(schema=1,model_id='robot-authoring-access-v1',source_roots=old['source_roots'],
        constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_robot_authoring_access_v1',symbol='run',source='experiments/run_robot_authoring_access_v1.py'),
        runner_config=cfg,files=inventory(ROOT,old['source_roots']))
    write_json(target,spec);pin=sha(target);verify(ROOT,target,pin)
    report=dict(status='prepared_unapproved',manifest=str(target),sha256=pin,sources=len(spec['files']),resources=len(resources),
        source_checkout=str(ROOT),registered=False,launched=False,author_calls_max=144,inspection_attempts_max=48,
        final_missions_max=512,new_control_updates=0,reused_control_updates=384,final_seed=73600000)
    write_json(ROOT/'docs/wspl/robot_access_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
